# CareCompass Architecture

CareCompass matches patients with healthcare providers using a pipeline of three
specialized AI agents coordinated by a LangGraph state machine. This document
explains how the pieces fit together and the design rules the system follows.

## Contents

- [System overview](#system-overview)
- [Data Gatherer](#1-data-gatherer-agent)
- [Preference Scorer](#2-preference-scorer-agent)
- [Critic Validator](#3-critic-validator-agent)
- [Orchestrator](#4-orchestrator)
- [Caching layer](#caching-layer)
- [Transparency surfaces](#transparency-surfaces)
- [Configuration](#configuration)
- [Design principles](#design-principles)

## System overview

```
User Input → Orchestrator → Data Gatherer → Preference Scorer → Critic Validator → Final Results
                 ↓
          Vector Store (ChromaDB) ← enrichment cache, encrypted at rest
```

Three different models fill three different roles, on purpose:

| Role | Model | Why |
|------|-------|-----|
| Extraction | Claude Haiku 4.5 | Fast, cost-effective reading of many web-page excerpts |
| Judge | GPT-5.6 Terra | Rubric-scored evaluation with anchored bands and cited evidence |
| Critic | Claude Opus 4.8 | Deep validation — deliberately a *different model family* than the judge it audits, so the check is independent |

State flows through a typed `WorkflowState` (a `TypedDict`): each agent receives
the accumulated state and adds its results (`gathered_data` →
`scored_providers` → `validation_results` → `final_recommendations`). Agents are
constructed through factory functions (`create_data_gatherer()`, etc.) and keep
their logic in their own modules under `agents/`.

## 1. Data Gatherer Agent

`agents/data_gatherer.py` — finds candidate providers on the live web and
enriches each one with verifiable evidence.

### Discovery

- **Multi-query search.** Three differently-phrased Tavily queries (professional
  directory, "top N" listicle, review-site-targeted) are fanned out for the
  user's city, merged, and deduplicated by URL. A single phrasing finds fewer
  distinct providers than three.
- **Adaptive ring expansion.** If the deduplicated home-city pool can't fill the
  research budget, the search expands to the nearest cities (computed from
  vendored geo data, never guessed). The expansion is recorded — how many
  candidates it added, how many were researched, and how many reached the
  final recommendations — so its cost/benefit is measurable, not just its
  firing.
- **Extraction.** Claude Haiku 4.5 extracts structured provider records from
  page excerpts. Pages are independent, so extraction runs as two concurrent
  calls over a partition of the pages when the page count justifies it.

### Excerpting

Raw pages are large; the extractor reads **anchor-centered windows** built by
`utils/excerpt.py`: boilerplate is stripped, then windows are centered on
anchor terms (specialty and review vocabulary for discovery; the provider's
name and per-platform header hints for enrichment). Discovery and enrichment
use different excerpt budgets because they read different kinds of documents —
discovery reads many-name directory pages, enrichment reads one provider's
profile, where the page header (rating, review count, years of experience)
gets a reserved slice of the budget. Anchors match on word boundaries with
careful handling of short surnames and plural forms ("reviews", "ratings"),
which real platform pages use for exactly the numbers the scorer needs.

### Enrichment

After core scoring, every provider inside the research budget that the cache
didn't serve gets **one platform-restricted search** across five independent
patient-review platforms (Healthgrades, Vitals, Zocdoc, WebMD, RateMDs).
Result slots are spent round-robin so every platform contributes a page before
any contributes a second, and each platform's slot prefers the provider's own
**profile page** over directory listings.

- **Identity is enforced in code.** `_observation_is_same_person` checks the
  page's own stated name against the target with a token-overlap threshold;
  the specialty rides in the extraction prompt as identity evidence but never
  in the search query (asserting it pulled directory pages above the
  provider's own profile).
- **One voice per platform.** Observations collapse to a single rating+count
  pair per platform domain, so a same-domain duplicate can never pose as a
  second opinion. Sources are classified as profile / listing / unknown
  (`utils/provenance.py`), and that classification breaks ties.
- **Provenance everywhere.** Review and insurance claims carry their source
  URLs; when a list has no source, the UI says so instead of hiding the row.
- **Outcomes are explicit.** Every provider records an `enrichment_outcome`
  (`cached` / `enriched` / `no_profile_found` / `identity_rejected` /
  `over_budget` / `failed`), so "we looked and found nothing" is
  distinguishable from "we never looked" — and the scorer treats missing data
  as missing, not as bad.

### Location evidence

Distances are computed **in code** (`utils/geo.py`): vendored GeoNames ZIP and
city centroids plus haversine. The LLM never estimates a distance. Each
distance records its precision (`zip` / `city`), because a city-centroid
distance is one shared coordinate for every provider in that city — the scorer
adds an uncertainty margin to those and never invents a spread it can't
justify. When neither point resolves, scoring falls back to honest tiers
(same ZIP / same city / same state / different).

## 2. Preference Scorer Agent

`agents/preference_scorer.py` — two layers, deterministic and AI, blended.

### Deterministic core (70% of the final score)

A weighted 0–100 score over three dimensions, with the user's Low/Medium/High
preferences normalized to weights:

- **Rating** — Bayesian-shrunk by review volume toward a prior, so a 5.0 from
  three reviews doesn't beat a 4.7 from four hundred. When two or more
  platforms agree, a count-weighted cross-platform blend replaces the single
  headline number. Unrated providers score the prior itself — unknown is
  *unknown*, not bad.
- **Location** — precedence: code-computed distance → page-stated distance →
  tier fallback. Tier values sit at the pessimistic edge of their band so a
  measured provider is never out-scored by an imputed one.
- **Experience** — points per year with a knee (diminishing returns after 15
  years) and a cap below what a strong measured rating reaches, so unverified
  tenure can't out-swing measured patient experience. Missing tenure scores
  the equivalent of a solid mid-career value rather than a penalty.

`score_core()` exposes this LLM-free ranking; the orchestrator uses it to
decide which providers earn enrichment.

### Rubric judge (30% of the final score)

GPT-5.6 Terra scores each provider 0–100 against a fixed rubric: review
substance (50), red flags & consistency (30), practical access (20). The
rubric's bands are **anchored prose that tiles the whole range** — a score no
anchor describes is one the model improvises, so there are explicit rungs for
absence of evidence, for mixed evidence, and for negative evidence. Absence is
never penalized; the same signal is never charged to two criteria.

Robustness mechanics:

- Provider order is shuffled; each record carries an index the response must
  echo, with name cross-checks, so a misbound entry can't silently overwrite
  another provider's rubric.
- The pool may be scored in two concurrent calls (config-gated), dealt
  round-robin so each call sees a representative mix.
- The completion token budget scales with pool size, truncation is detected,
  and partial responses are salvaged entry-by-entry.
- The judge reads each provider's full review summary under a shared bound
  that breaks on a word and leaves a visible ellipsis — a silently truncated
  summary would bias exactly the criteria whose evidence lives in its tail.

`final_score = 0.7 × core + 0.3 × judge`, a true 0–100 scale.

## 3. Critic Validator Agent

`agents/critic_validator.py` — Claude Opus 4.8, two parallel calls per search.

- **Bias analysis** reads the whole ordering plus each dimension's weighted
  contribution to it, so causal claims ("X is ranked first because…") must
  cite arithmetic rather than guess. It returns two registers: a
  patient-facing explanation and a technical one for the developer surfaces.
- **Deep validation** reviews each provider against fixed verdict criteria
  (approve / conditional / reject, with cited evidence and a confidence tied
  to how much platform evidence exists). Verdicts are per-provider, so the
  pool is split across two concurrent calls, dealt so each call still sees a
  representative spread.
- **The critic audits the judge.** It receives the judge's rubric verbatim
  (imported from the scorer module, so the two can't drift) plus the judge's
  scores and citations, and reports citations the evidence doesn't support.
  Judge mistakes are reported to the developer surfaces and never charged to
  the provider's score.
- **Findings feed back deterministically.** `refine_rankings()` folds verdicts
  into the final order as pure post-processing — no additional LLM calls. Every
  adjustment is recorded per provider, and verdicts bind to providers by name
  tokens with a high threshold, never by list position.

A recommendation asserts three completions — researched, judged, critiqued —
and `withheld_reason()` verifies all three. The shortlist may be shorter than
five; a provider that can't be fully vouched for is listed with its reason
rather than silently carded.

## 4. Orchestrator

`agents/orchestrator.py` — a LangGraph workflow with typed state, live
progress callbacks for the UI, error tracking in `WorkflowState`, retry
logic, and per-search cost accounting (`utils/cost_tracker.py` — a
thread-safe accumulator whose summary feeds the UI cost card).

The orchestrator pins the research budget (`MAX_PROVIDERS_TO_ENRICH`) before
enrichment runs; enrichment, the judge, and the critic all honor the same cut.
Providers past it are marked `over_budget`, stay ranked, and reach no model —
one budget, applied once, visible everywhere.

## Caching layer

`utils/vector_store.py` — ChromaDB with payloads encrypted at rest
(single-key Fernet, `utils/encryption.py`).

- **Identity-keyed, never similarity-served.** Cache reads are a keyed lookup
  on `provider_key` (SHA-256 of normalized name + city). Two neurologists in
  one city embed almost identically, so nearest-neighbour lookup could serve
  one physician's reviews for another; embeddings are stored for future
  features but never decide identity.
- **Substantive-data guard.** Only providers with real evidence are cached —
  extractor placeholders ("No reviews available") don't count, so a provider
  whose enrichment found nothing isn't frozen into the cache for the TTL.
- **Warm must reproduce cold.** Distances and scores are never cached (they
  depend on the user's location); everything derived from cached observations
  is recomputed on every hit through the same code paths a cold run uses.
- **Stable keys.** The cache key is pinned before enrichment runs, because
  enrichment can sharpen a provider's address and a key computed afterwards
  would never be found by a later read.
- A TTL (`PROVIDER_CACHE_TTL_DAYS`) bounds staleness, a schema version gates
  incompatible rows, and cache hits union with — never replace — evidence the
  current run already found.

## Transparency surfaces

The Streamlit UI (`app.py`, themed by `utils/theme.py`) treats explanation as
a first-class output:

- **Provider cards** — match ring, "why this match" callout, at-a-glance chips
  (network check, pool superlatives, critic verdict, sentiment, caveats), and
  an AI-analysis expander showing both score layers, judge-attributed
  strengths/considerations with citations, and the critic's independent
  review. Every rubric row shows either a citation or a plain note saying what
  wasn't found.
- **Other providers considered** — everything below the shortlist, grouped by
  why: ranked below the cut, researched but not recommendable (with the
  reason), or past the research budget (scores labeled provisional).
- **Agent Decision Process** — names every withheld provider and the stage
  that didn't complete, with the system's own failures listed first.
- **Responsible-AI panel** — bias check and red-flag tiles, the bias
  explanation (labeled with the ordering it describes), a judge-consistency
  note (count only — internal rubric vocabulary stays off patient surfaces),
  and "What this ranking doesn't capture".
- **Cost card & timeline** — per-search token/credit estimates and a
  step-by-step timing table attributed to the agent that actually did the
  work.
- **FHIR network check** (`fhir/verify.py`) — an optional sidebar
  verification of insurance-network membership against a FHIR directory
  (sandbox by default). It never affects scores; insurance data from
  directories is displayed as unverified and never rides in a search query.

## Configuration

`utils/config.py` reads everything from the environment (`.env` locally). The
knobs that shape a run:

| Knob | Default | Effect |
|------|---------|--------|
| `GATHERER_MODEL` / `JUDGE_MODEL` / `CRITIC_MODEL` | Haiku 4.5 / GPT-5.6 Terra / Opus 4.8 | Per-role model selection |
| `MAX_PROVIDERS_TO_ENRICH` | 8 | The research budget — one cut honored by enrichment, judge, and critic; also the shortlist's headroom for coverage failures |
| `ENRICHMENT_MAX_WORKERS` | 8 | Enrichment concurrency (sets the number of waves, not the amount of work) |
| `JUDGE_PARALLEL_ENABLED` | true | Whether the judge scores the pool in two concurrent calls |
| `MULTI_QUERY_ENABLED` / `MIN_CANDIDATE_POOL` / `MAX_RING_CITIES` | true / 8 / 2 | Discovery breadth and when ring expansion fires |
| `PROVIDER_CACHE_TTL_DAYS` | 7 | Cache freshness window (0 disables reuse without discarding data) |
| `TAVILY_SEARCH_DEPTH` | basic | Search depth; platform-targeted searches always run advanced |

See [`.env.example`](../.env.example) for the full list, including auth,
encryption, rate limiting, and TLS settings.

## Design principles

1. **Measured beats imputed.** A provider with real evidence must never be
   out-scored by one with a flattering guess; imputations sit at the
   pessimistic edge of their band.
2. **Absence is labeled, never penalized.** Missing data reflects the
   system's coverage, not the provider's quality — unknowns score neutral
   values stated as such.
3. **Identity is enforced in code.** Name-overlap checks and keyed cache
   lookups decide who's who; embeddings and LLM judgment never do.
4. **Every number carries provenance.** Ratings, insurance lists, and
   distances record where they came from and how precise they are.
5. **Failures are visible.** Withheld providers, truncated model responses,
   over-budget cuts, and cache decisions all surface in the UI with reasons —
   the correct response to an unvouchable recommendation is a shorter
   shortlist, not a quiet one.
6. **The checker must be independent.** The critic reads the same evidence as
   the judge (a tested payload-parity contract) but runs in a different model
   family, and its findings about the judge never move provider scores.
