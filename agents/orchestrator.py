"""LangGraph Orchestrator for coordinating the multi-agent healthcare provider matching workflow."""

import logging
import time
from typing import Dict, List, Any, Optional, TypedDict, Callable
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from datetime import datetime

from .data_gatherer import DataGathererAgent
from .preference_scorer import PreferenceScorerAgent
from .critic_validator import CriticValidatorAgent, refine_rankings
from utils.vector_store import get_vector_store
from utils.config import get_config
from utils.cost_tracker import get_cost_tracker
from utils.provenance import url_page_kind

logger = logging.getLogger(__name__)


# Enrichment outcomes that mean we actually HOLD this provider's details.
# `_classify_enrichment` returns "enriched" only when an identity-accepted
# review observation or a real (non-placeholder) summary was obtained, and the
# cache refuses to store placeholder rows — so these two are the outcomes where
# a card has something true to show.
_OUTCOMES_WITH_DATA = frozenset({"enriched", "cached"})

# Why a provider was kept out of the shortlist. The first three are coverage —
# nobody's fault, and normal operation. The last two are OUR pipeline failing on
# a provider whose data we successfully found, which is a different claim and
# belongs on a different surface.
_WITHHELD_LABELS = {
    "over_budget": "not researched — outside this search's research budget",
    "no_profile_found": "researched, but no reviews were found",
    "identity_rejected": "reviews found, but not verifiably this provider's",
    "failed": "the review lookup failed",
    "not_judged": "our rubric scoring did not complete for this provider",
    "not_critiqued": "our independent review did not complete for this provider",
}

# The subset above that is a fault in OUR pipeline rather than a gap in what the
# web holds. Separated because the Responsible-AI panel reports our own errors,
# and conflating the two would tell a patient that a provider was unverifiable
# when in fact we simply failed to score them.
_PIPELINE_FAILURE_REASONS = frozenset({"not_judged", "not_critiqued"})


def withheld_reason(provider: Dict[str, Any]) -> Optional[str]:
    """Why this provider cannot be a recommendation, or None if it can.

    Order matters: the earliest unmet stage is the reason. A provider that was
    never researched is not ALSO "not judged" — reporting the downstream
    symptom would blame our pipeline for a cut we made deliberately.
    """
    outcome = str(provider.get("enrichment_outcome") or "")
    if outcome not in _OUTCOMES_WITH_DATA:
        # "" (never enriched at all) falls here too, and is reported as the
        # budget cut, which is the only way it arises.
        return outcome if outcome in _WITHHELD_LABELS else "over_budget"
    if not (provider.get("ai_rubric") or {}):
        return "not_judged"
    if not provider.get("critic_review"):
        return "not_critiqued"
    return None


def _is_recommendable(provider: Dict[str, Any]) -> bool:
    return withheld_reason(provider) is None


def _withheld_summary(withheld: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts by reason, with our own failures counted separately."""
    by_reason: Dict[str, int] = {}
    pipeline_failure_names: List[str] = []
    for provider in withheld:
        reason = withheld_reason(provider) or "unknown"
        by_reason[reason] = by_reason.get(reason, 0) + 1
        if reason in _PIPELINE_FAILURE_REASONS:
            pipeline_failure_names.append(str(provider.get("name", "Unknown")))
    return {
        "total": len(withheld),
        "by_reason": by_reason,
        "pipeline_failures": len(pipeline_failure_names),
        "pipeline_failure_names": pipeline_failure_names,
        "no_data": sum(
            count for reason, count in by_reason.items()
            if reason in ("no_profile_found", "identity_rejected", "failed")
        ),
        "not_researched": by_reason.get("over_budget", 0),
    }


class ProgressUpdate(TypedDict):
    """Real-time progress update structure."""
    step_name: str           # "gather_data", "score_providers", etc.
    agent_name: str          # "DataGathererAgent", "PreferenceScorerAgent", etc.
    status: str              # "started", "in_progress", "completed", "failed"
    action: str              # "Found 7 providers in Phoenix, AZ"
    metrics: Dict[str, Any]  # {"providers_found": 7, "location": "Phoenix, AZ"}
    progress_percentage: int # 0-100
    timestamp: str


ProgressCallback = Callable[[ProgressUpdate], None]


class WorkflowState(TypedDict):
    """State structure for the provider matching workflow."""
    # Input parameters
    specialty: str
    location: str
    insurance: Optional[str]
    preferences: Dict[str, Any]
    # Per-search, not per-orchestrator: get_orchestrator() is cached on
    # (fhir_enabled, fast_demo), so routing the cache flag through construction
    # would rebuild the whole orchestrator every time the sidebar toggled.
    use_cache: bool

    # Agent outputs
    gathered_data: Dict[str, Any]
    scored_providers: Dict[str, Any]
    validation_results: Dict[str, Any]

    # Workflow metadata
    current_step: str
    workflow_id: str
    error_messages: List[str]
    execution_log: List[Dict[str, Any]]

    # Final output
    final_recommendations: List[Dict[str, Any]]
    workflow_summary: Dict[str, Any]


class ProviderMatchingOrchestrator:
    """LangGraph orchestrator for coordinating all agents in the provider matching workflow."""

    def __init__(self, progress_callback: Optional[ProgressCallback] = None):
        """Initialize the orchestrator with all agents and workflow components."""
        self.config = get_config()
        self.progress_callback = progress_callback
        self._step_started_at: Dict[str, float] = {}

        # Initialize agents
        self.data_gatherer = DataGathererAgent()
        self.preference_scorer = PreferenceScorerAgent()
        self.critic_validator = CriticValidatorAgent()
        self.vector_store = get_vector_store()

        # Build workflow graph
        self.workflow = None
        self._build_workflow()

        logger.info("Provider matching orchestrator initialized")

    def _build_workflow(self) -> None:
        """Build the LangGraph workflow for provider matching."""
        try:
            # Create state graph
            workflow = StateGraph(WorkflowState)

            # Add workflow nodes
            workflow.add_node("initialize", self._initialize_workflow)
            workflow.add_node("gather_data", self._gather_provider_data)
            workflow.add_node("score_providers", self._score_providers)
            workflow.add_node("validate_rankings", self._validate_rankings)
            workflow.add_node("finalize_results", self._finalize_results)
            workflow.add_node("handle_error", self._handle_error)

            # Set entry point
            workflow.set_entry_point("initialize")

            # Add edges (workflow flow)
            workflow.add_edge("initialize", "gather_data")
            workflow.add_conditional_edges(
                "gather_data",
                self._check_data_gathering_success,
                {
                    "success": "score_providers",
                    "error": "handle_error"
                }
            )
            workflow.add_conditional_edges(
                "score_providers",
                self._check_scoring_success,
                {
                    "success": "validate_rankings",
                    "error": "handle_error"
                }
            )
            workflow.add_conditional_edges(
                "validate_rankings",
                self._check_validation_success,
                {
                    "success": "finalize_results",
                    "error": "handle_error"
                }
            )
            workflow.add_edge("finalize_results", END)
            workflow.add_edge("handle_error", END)

            # Compile workflow
            self.workflow = workflow.compile()
            logger.info("LangGraph workflow compiled successfully")

        except Exception as e:
            logger.error(f"Failed to build workflow: {e}")
            raise

    def _log_step(
        self,
        state: WorkflowState,
        step: str,
        status: str,
        details: Dict[str, Any],
        nested_s: float = 0.0,
    ) -> None:
        """Log workflow step execution with real timestamps and durations.

        `nested_s` is wall clock this step spent inside a DIFFERENT step that
        reports its own row on the timeline, and it is subtracted here so the
        rows sum to the run instead of double-counting. Exactly one caller uses
        it: `score_providers` wraps review enrichment, ~54s of a 145s run on
        2026-07-28, and the timeline attributed all of it to the preference
        scorer — whose own deterministic work is microseconds and whose LLM call
        is a fifth of the step. The banner text was corrected for this same
        confusion in an earlier round; the CLOCK was not.
        """
        now = time.perf_counter()
        if status == "started":
            self._step_started_at[step] = now
        elif status in ("completed", "failed") and step in self._step_started_at:
            elapsed = now - self._step_started_at[step] - max(nested_s, 0.0)
            details = {**details, "elapsed_s": round(max(elapsed, 0.0), 2)}

        log_entry = {
            "step": step,
            "status": status,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "details": details
        }
        state["execution_log"].append(log_entry)
        logger.info(f"Workflow step completed: {step} - {status}")

    def _emit_progress(
        self,
        step_name: str,
        agent_name: str,
        status: str,
        action: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Emit progress update to registered callback."""
        if not self.progress_callback:
            return

        # Progress mapping: 5 steps total (initialize, gather, score, validate, finalize)
        step_progress_map = {
            "initialize": {"started": 5, "completed": 10},
            "gather_data": {"started": 10, "in_progress": 25, "completed": 35},
            "score_providers": {"started": 35, "in_progress": 50, "completed": 60},
            "validate_rankings": {"started": 60, "in_progress": 75, "completed": 85},
            "finalize_results": {"started": 85, "completed": 100}
        }

        progress_pct = step_progress_map.get(step_name, {}).get(status, 0)

        update = ProgressUpdate(
            step_name=step_name,
            agent_name=agent_name,
            status=status,
            action=action,
            metrics=metrics,
            progress_percentage=progress_pct,
            timestamp=datetime.now().isoformat()
        )

        try:
            self.progress_callback(update)
        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")
            # Don't let callback failures break workflow

    def _initialize_workflow(self, state: WorkflowState) -> WorkflowState:
        """Initialize the workflow with input validation and setup."""
        try:
            # Fresh cost/timing accounting for this run (the orchestrator is
            # cached across searches, so per-run state must reset here)
            get_cost_tracker().reset()
            self._step_started_at = {}

            self._log_step(state, "initialize", "started", {"input_validation": "in_progress"})

            # Validate required inputs
            if not state.get("specialty") or not state.get("location"):
                raise ValueError("Specialty and location are required")

            # Set default preferences if not provided
            if not state.get("preferences"):
                state["preferences"] = {
                    "location_weight": 0.4,
                    "rating_weight": 0.3,
                    "insurance_priority": 0.3
                }

            # Initialize workflow metadata
            state["current_step"] = "initialize"
            state["error_messages"] = []
            state["execution_log"] = []

            # Generate workflow ID
            import uuid
            state["workflow_id"] = str(uuid.uuid4())[:8]

            self._log_step(state, "initialize", "completed", {
                "workflow_id": state["workflow_id"],
                "specialty": state["specialty"],
                "location": state["location"]
            })

            return state

        except Exception as e:
            state["error_messages"].append(f"Initialization failed: {str(e)}")
            self._log_step(state, "initialize", "failed", {"error": str(e)})
            return state

    def _gather_provider_data(self, state: WorkflowState) -> WorkflowState:
        """Execute data gathering using the DataGathererAgent."""
        try:
            state["current_step"] = "gather_data"
            self._log_step(state, "gather_data", "started", {
                "agent": "DataGathererAgent",
                "search_params": {
                    "specialty": state["specialty"],
                    "location": state["location"],
                    "insurance": state.get("insurance")
                }
            })

            # Emit progress: started
            self._emit_progress(
                step_name="gather_data",
                agent_name="DataGathererAgent",
                status="started",
                action=f"Searching for {state['specialty']} providers in {state['location']}",
                metrics={"specialty": state["specialty"], "location": state["location"]}
            )

            # Gather provider data. enrich=False: review enrichment runs in
            # the scoring step instead, after core ranking, so its budget
            # targets the likely top-K rather than extraction order
            gathered_data = self.data_gatherer.gather_providers(
                specialty=state["specialty"],
                location=state["location"],
                insurance=state.get("insurance"),
                enrich=False
            )

            state["gathered_data"] = gathered_data

            # Emit progress: in progress (with FHIR details)
            provider_count = len(gathered_data.get("providers", []))
            search_meta = gathered_data.get("search_metadata", {})
            fhir_count = search_meta.get("fhir_count", 0)
            tavily_count = search_meta.get("tavily_count", 0)

            fhir_note = ""
            if fhir_count > 0:
                fhir_note = f" (verified {fhir_count} via FHIR)"

            self._emit_progress(
                step_name="gather_data",
                agent_name="DataGathererAgent",
                status="in_progress",
                action=f"Found {provider_count} providers in {state['location']}{fhir_note}",
                metrics={
                    "providers_found": provider_count,
                    "fhir_count": fhir_count,
                    "tavily_count": tavily_count,
                }
            )

            # The vector store is written AFTER enrichment now (see
            # data_gatherer._store_enrichment), not here. Writing pre-enrichment
            # providers stored the cheap half of a search under no timestamp,
            # and nothing ever read it back — one embedding call per search for
            # nothing.

            if gathered_data.get("status") != "success":
                state["error_messages"].append(f"Data gathering failed with status: {gathered_data.get('status')}")

            self._log_step(state, "gather_data", "completed", {
                "providers_found": provider_count,
                "fhir_verified": fhir_count,
                "tavily_found": tavily_count,
                "data_status": gathered_data.get("status"),
                "vector_store_updated": True
            })

            # Emit progress: completed
            self._emit_progress(
                step_name="gather_data",
                agent_name="DataGathererAgent",
                status="completed",
                action=f"Extracted {provider_count} providers successfully{fhir_note}",
                metrics={
                    "providers_found": provider_count,
                    "fhir_count": fhir_count,
                    "tavily_count": tavily_count,
                    "status": gathered_data.get("status")
                }
            )

            return state

        except Exception as e:
            state["error_messages"].append(f"Data gathering failed: {str(e)}")
            self._log_step(state, "gather_data", "failed", {"error": str(e)})
            return state

    def _score_providers(self, state: WorkflowState) -> WorkflowState:
        """Execute provider scoring using the PreferenceScorerAgent."""
        try:
            state["current_step"] = "score_providers"
            self._log_step(state, "score_providers", "started", {
                "agent": "PreferenceScorerAgent",
                "providers_to_score": len(state["gathered_data"].get("providers", []))
            })

            # Emit progress: started
            # Truthful timeline: this node runs core ranking -> enrichment ->
            # judge, so the start banner must not read as the judge already
            # working (field observation: "Terra analyzed before enrichment?")
            self._emit_progress(
                step_name="score_providers",
                agent_name="PreferenceScorerAgent",
                status="started",
                action=f"Ranking {len(state['gathered_data'].get('providers', []))} providers by your weights (deterministic core)",
                metrics={"providers_to_score": len(state["gathered_data"].get("providers", []))}
            )

            providers = state["gathered_data"]["providers"]
            preferences = state["preferences"]
            judge_preferences = dict(preferences)

            # Stage 1: deterministic core ranking (no LLM) decides who
            # deserves the research budget.
            core_ranked = self.preference_scorer.score_core(providers, preferences)

            # ONE cut, pinned here, honoured by every stage below.
            #
            # It used to gate enrichment alone: the judge and critic then ran
            # over the whole pool, scoring providers nobody had researched. That
            # is not a lenient ranking, it is a wrong one — a rubric applied to
            # an empty record grades OUR coverage and reports it as the
            # provider's quality, and the critic spent Opus tokens issuing
            # verdicts on the same emptiness. Enrichment is ~9% of a run's cost;
            # the judge and critic together are ~74%. Rationing the cheap stage
            # while the expensive ones ran wide was backwards in both senses.
            #
            # Pinned BEFORE enrichment because enrichment backfills ratings and
            # moves core scores — a set re-derived afterwards would not be the
            # set we actually researched.
            budget = self.config.MAX_PROVIDERS_TO_ENRICH
            selected = core_ranked[:budget]
            deferred = core_ranked[budget:]
            for provider in deferred:
                provider["enrichment_outcome"] = "over_budget"

            # Stage 2: read reviews for the selection. Every provider in it that
            # the cache didn't serve gets a live pass — no tiering within.
            #
            # Timed and logged as its OWN step. It is DataGathererAgent work —
            # a Tavily search and a Haiku extraction per provider — that happens
            # to run inside this node because the core ranking above decides who
            # gets it. On the timeline it was invisible, and "Preference Scoring
            # — 70.1s" told a reader the scorer was slow when the scorer's share
            # was ~15s of it.
            enrich_elapsed = 0.0
            if selected:
                self._emit_progress(
                    step_name="score_providers",
                    agent_name="DataGathererAgent",
                    status="in_progress",
                    action="Reading reviews for top candidates",
                    metrics={"enrichment_budget": len(selected)}
                )
                self._log_step(state, "enrich_reviews", "started", {
                    "agent": "DataGathererAgent",
                    "providers_to_enrich": len(selected),
                })
                enrich_started = time.perf_counter()
                self.data_gatherer.enrich_providers(
                    selected,
                    location=state["location"],
                    specialty=state["specialty"],
                    use_cache=state.get("use_cache", True),
                )
                enrich_elapsed = time.perf_counter() - enrich_started
                outcomes: Dict[str, int] = {}
                for provider in selected:
                    key = str(provider.get("enrichment_outcome") or "unknown")
                    outcomes[key] = outcomes.get(key, 0) + 1
                self._log_step(state, "enrich_reviews", "completed", {
                    "agent": "DataGathererAgent",
                    "outcomes": outcomes,
                    # Coverage of the numbers we ended up with, not just whether
                    # a search ran: a pair sourced only from a directory index
                    # is attributable to many doctors, not this one.
                    "profile_backed": sum(
                        1 for p in selected if p.get("profile_backed_platforms")
                    ),
                })

            # Stage 3: full scoring — core recomputed (enrichment can backfill
            # ratings), then the rubric judge reads the enriched evidence for
            # the selection only. `selected + deferred` preserves the pinned
            # order that `judge_count` indexes into.
            self._emit_progress(
                step_name="score_providers",
                agent_name="PreferenceScorerAgent",
                status="in_progress",
                action="Scoring the enriched evidence against the rubric",
                metrics={"providers_to_judge": len(selected)}
            )
            scored_results = self.preference_scorer.score_providers(
                providers=selected + deferred,
                preferences=judge_preferences,
                judge_count=len(selected),
            )

            state["scored_providers"] = scored_results

            # A SOFT failure — the agent caught its own exception and returned
            # status "error" without raising — routed to handle_error, which
            # only READS error_messages. Nothing appended, so the run reported
            # success=True with an empty recommendation list and the UI told
            # the user to adjust their search criteria. `_gather_provider_data`
            # already appends here; scoring and validation did not.
            if scored_results.get("status") != "success":
                state["error_messages"].append(
                    f"Scoring failed with status: {scored_results.get('status')}"
                )

            self._log_step(state, "score_providers", "completed", {
                "ranking_status": scored_results.get("status"),
                "providers_ranked": len(scored_results.get("ranked_providers", [])),
                "top_provider": scored_results.get("scoring_metadata", {}).get("top_provider")
            }, nested_s=enrich_elapsed)

            # Emit progress: completed
            self._emit_progress(
                step_name="score_providers",
                agent_name="PreferenceScorerAgent",
                status="completed",
                action=f"Ranked {len(scored_results.get('ranked_providers', []))} providers",
                metrics={
                    "providers_ranked": len(scored_results.get("ranked_providers", [])),
                    "top_provider": scored_results.get("scoring_metadata", {}).get("top_provider", "Unknown")
                }
            )

            return state

        except Exception as e:
            state["error_messages"].append(f"Provider scoring failed: {str(e)}")
            self._log_step(state, "score_providers", "failed", {"error": str(e)})
            return state

    def _validate_rankings(self, state: WorkflowState) -> WorkflowState:
        """Execute ranking validation using the CriticValidatorAgent."""
        try:
            state["current_step"] = "validate_rankings"

            # The critic audits exactly what the judge scored. A provider the
            # judge never saw has no rubric to check and no researched evidence
            # to argue from, so a verdict on them would be the critic reviewing
            # our data gap. This narrows a rule the register states broadly —
            # "the critic validates every ranked provider" — and the reasoning
            # behind that rule (a signal about our own scoring must not stop at
            # rank 5) still holds in full inside the judged set.
            audited = [
                provider for provider in state["scored_providers"].get("ranked_providers", [])
                if provider.get("ai_judged") is not False
            ]

            self._log_step(state, "validate_rankings", "started", {
                "agent": "CriticValidatorAgent",
                "providers_to_validate": len(audited)
            })

            # Emit progress: started
            self._emit_progress(
                step_name="validate_rankings",
                agent_name="CriticValidatorAgent",
                status="started",
                action="Validating provider rankings for bias",
                metrics={"providers_to_validate": len(audited)}
            )

            validation_results = self.critic_validator.validate_rankings(
                ranked_providers=audited,
                preferences=dict(state["preferences"])
            )

            state["validation_results"] = validation_results

            if validation_results.get("status") != "success":
                state["error_messages"].append(
                    f"Validation failed with status: {validation_results.get('status')}"
                )

            self._log_step(state, "validate_rankings", "completed", {
                "validation_status": validation_results.get("status"),
                "bias_severity": validation_results.get("validation_metadata", {}).get("bias_severity"),
                "ranking_confidence": validation_results.get("validation_metadata", {}).get("ranking_confidence")
            })

            # Emit progress: completed
            confidence = validation_results.get("validation_metadata", {}).get("ranking_confidence", "medium")
            self._emit_progress(
                step_name="validate_rankings",
                agent_name="CriticValidatorAgent",
                status="completed",
                action=f"Validation complete with {confidence} confidence",
                metrics={
                    "confidence": confidence,
                    "bias_severity": validation_results.get("validation_metadata", {}).get("bias_severity", "unknown")
                }
            )

            return state

        except Exception as e:
            state["error_messages"].append(f"Ranking validation failed: {str(e)}")
            self._log_step(state, "validate_rankings", "failed", {"error": str(e)})
            return state

    def _finalize_results(self, state: WorkflowState) -> WorkflowState:
        """Finalize the workflow results and prepare final recommendations."""
        try:
            state["current_step"] = "finalize_results"
            self._log_step(state, "finalize_results", "started", {
                "consolidating_results": True
            })

            # Extract final recommendations
            ranked_providers = state["scored_providers"].get("ranked_providers", [])
            validation_results = state["validation_results"].get("validation_results", {})

            # Close the critique loop: fold the critic's findings (statuses,
            # red flags, confidence) back into the ranking; alternative
            # scenarios are information-only and never move scores.
            # Pure post-processing — no extra LLM calls, no added latency.
            refined_providers, refinement_summary = refine_rankings(
                ranked_providers, state["validation_results"]
            )

            # The shortlist is drawn from providers we ACTUALLY RESEARCHED.
            #
            # `over_budget` providers were never enriched, judged or validated,
            # so their score is built entirely from imputations — the rating
            # prior, the unknown-tenure constant, and a city centroid shared by
            # everyone in the city. On the 2026-07-27 run that produced four
            # providers at an identical 64, and they outranked three fully
            # researched ones at 60/58/57. The gap was not quality: it was the
            # critic's -8 "conditional" penalty, which only a provider the
            # critic SAW can receive. A researched provider can only lose points
            # at refinement; an unresearched one has adjustment 0 by
            # construction, so the comparison is between a docked score and an
            # undocked one.
            #
            # A RECOMMENDATION requires all three stages to have completed for
            # that provider — we found their details, the judge scored them, and
            # the critic reviewed them. Anything less produces a card whose
            # reviews, rubric or verdict is blank, presented as a top result.
            #
            # Note `ai_judged` is NOT the judged test. It is set to False only on
            # providers deferred past the budget; a provider that WAS submitted
            # to the judge but whose entry the judge omitted (truncation, parse
            # failure, name mismatch) keeps it unset and receives ai_score 50.0
            # from the setdefault in `score_providers`. So `ai_judged is not
            # False` means "we sent them", while a non-empty `ai_rubric` means
            # "the judge actually scored them" — which is the claim a card makes.
            recommendable = [p for p in refined_providers if _is_recommendable(p)]
            withheld = [p for p in refined_providers if not _is_recommendable(p)]

            # No "fill from unresearched if the shortlist is short" branch, and
            # none is reachable. `over_budget` is set only on core_ranked[budget:]
            # (see the enrichment stage above), so len(researched) is
            # min(pool, budget) and unresearched is non-empty only when
            # pool > budget. Needing a fill means min(pool, budget) < 5 AND
            # pool > budget, which reduces to budget < 5 — the two conditions
            # are otherwise mutually exclusive. A pool under 5 has nobody to
            # fill FROM, because everyone fit inside the budget.
            #
            # And where it IS reachable — an operator setting
            # MAX_PROVIDERS_TO_ENRICH to 3 — padding would be wrong on purpose:
            # it would put providers we explicitly declined to research onto
            # recommendation cards. Three researched providers means three cards.
            shortlist = recommendable[:5]

            # Everything not shortlisted, recommendable first. The UI groups on
            # `withheld_reason`; a null reason means the provider is a valid
            # recommendation that simply ranked below the shortlist.
            remainder = recommendable[5:] + withheld

            shortfall = _withheld_summary(withheld)
            if shortfall["total"]:
                logger.info(
                    "Withheld %d provider(s) from the shortlist: %s",
                    shortfall["total"],
                    ", ".join(f"{k}={v}" for k, v in sorted(shortfall["by_reason"].items())),
                )
            if shortfall["pipeline_failures"]:
                # OUR failure, not the provider's — a stage we paid for did not
                # produce output for a provider whose data we successfully found.
                logger.warning(
                    "%d provider(s) were fully researched but withheld because our "
                    "own scoring did not complete for them: %s",
                    shortfall["pipeline_failures"],
                    ", ".join(shortfall["pipeline_failure_names"]),
                )

            # Prepare final recommendations from the refined order
            final_recommendations = []
            top_validations = validation_results.get("top_provider_validation", {}).get("top_provider_validations", [])
            for i, provider in enumerate(shortlist):
                recommendation = {
                    "rank": i + 1,
                    "provider": provider,
                    "recommendation_confidence": "high",  # Could be adjusted based on validation
                    "ai_reasoning": provider.get("ai_reasoning", ""),
                    "validation_notes": ""
                }

                # Validation entries reference the scorer's pre-refinement ranks
                for val in top_validations:
                    if val.get("rank") == provider.get("pre_refinement_rank"):
                        recommendation["validation_notes"] = val.get("validation_notes", "")
                        recommendation["recommendation_confidence"] = val.get("confidence_in_recommendation", "medium")
                        break

                final_recommendations.append(recommendation)

            state["final_recommendations"] = final_recommendations

            # Everything below the shortlist — a compact, log-safe shape for the
            # UI's "Other providers considered" expander (transparency into what
            # didn't make the cut). Researched entries come first; the UI groups
            # on `researched` and gives each group its own heading.
            other_providers = [
                {
                    "rank": offset + len(shortlist) + 1,
                    # Null when the provider is a valid recommendation that
                    # simply ranked below the shortlist. Otherwise the earliest
                    # stage that did not complete — which is what the UI groups
                    # on, and what the developer surface prints per provider.
                    "withheld_reason": withheld_reason(provider),
                    "withheld_label": _WITHHELD_LABELS.get(
                        withheld_reason(provider) or "", ""
                    ),
                    # Kept for the score-scale distinction: an `over_budget`
                    # provider reached NO model, so its number is pure
                    # imputation, while a researched-but-unrecommendable one was
                    # judged and could be docked by the critic.
                    "researched": provider.get("enrichment_outcome") != "over_budget",
                    "name": provider.get("name", "Unknown"),
                    "specialty": provider.get("specialty", ""),
                    "rating": provider.get("rating"),
                    "review_count": provider.get("review_count"),
                    "blended_rating": provider.get("blended_rating"),
                    "blended_review_count": provider.get("blended_review_count"),
                    "blended_platform_count": provider.get("blended_platform_count"),
                    "computed_distance_miles": provider.get("computed_distance_miles"),
                    # Without precision a shared city centroid is
                    # indistinguishable from a measured distance — the honesty
                    # fix round 7 shipped for the top cards, which never
                    # reached this projection.
                    "distance_precision": provider.get("distance_precision"),
                    "location_match": provider.get("location_match"),
                    # Distinguishes "we looked and found nothing" from "we
                    # never looked", so an "Unrated" row can say which.
                    "enrichment_outcome": provider.get("enrichment_outcome"),
                    "critic_status": (provider.get("critic_review") or {}).get("status"),
                    "final_score": provider.get("final_score"),
                    "refined_score": provider.get("refined_score"),
                }
                for offset, provider in enumerate(remainder)
            ]

            # Per-provider review-coverage diagnostic, for the developer surface
            # only (the "Data Gatherer" tab). Kept OUT of the recommendation and
            # `other_providers` shapes because it answers a question about OUR
            # pipeline, not about the provider — the same split the withheld
            # reasons use.
            #
            # It exists because three different failures produce the identical
            # finished card: the platform's profile was never returned by the
            # search, it was returned but yielded no observation, or it yielded
            # one that lost the same-domain collapse to a directory listing. On
            # 2026-07-28 two providers carded "healthgrades.com — listing page"
            # as their best single source and nothing in the run said which of
            # the three had happened.
            # What ring expansion actually bought, measured at the only place
            # it matters. `search_metadata.ring_added` says how many candidates
            # it contributed; this says how far they got.
            #
            # The decision it informs is whether MIN_CANDIDATE_POOL should
            # drop below the research budget so the
            # ring stops firing on nearly every search. Candidates added is the
            # wrong number for that: the ring's real cost is that it FILLS the
            # budget, so every provider it adds also consumes an enrichment
            # search, a slot in the judge prompt and an Opus verdict. If those
            # providers never reach a card, that is spend for nothing; if they
            # routinely take cards, the ring is carrying the results and the
            # threshold should stay. Nothing distinguished those two cases.
            shortlist_names = {p.get("name") for p in shortlist}
            ring_contribution = {
                "added": sum(
                    1 for p in refined_providers
                    if p.get("discovery_source") == "ring"
                ),
                "researched": sum(
                    1 for p in refined_providers
                    if p.get("discovery_source") == "ring"
                    and p.get("enrichment_outcome") != "over_budget"
                ),
                "shortlisted": sum(
                    1 for p in refined_providers
                    if p.get("discovery_source") == "ring"
                    and p.get("name") in shortlist_names
                ),
            }

            review_coverage = [
                {
                    "name": provider.get("name", "Unknown"),
                    "discovery_source": provider.get("discovery_source"),
                    "outcome": provider.get("enrichment_outcome"),
                    "platform_pairs": provider.get("platform_pair_count", 0),
                    "profile_backed_platforms": provider.get("profile_backed_platforms", 0),
                    "headline_source": provider.get("review_source_url"),
                    "headline_kind": url_page_kind(provider.get("review_source_url")),
                    # Absent on a cache hit, which ran no search — see
                    # `_enrich_one`. An empty list would read as "we looked and
                    # found nothing".
                    "sources": provider.get("enrichment_sources"),
                }
                for provider in refined_providers
                if provider.get("enrichment_outcome") != "over_budget"
            ]

            # Create workflow summary
            workflow_summary = {
                "workflow_id": state["workflow_id"],
                "total_providers_found": len(state["gathered_data"].get("providers", [])),
                "final_recommendations_count": len(final_recommendations),
                "other_providers": other_providers,
                "review_coverage": review_coverage,
                "ring_contribution": ring_contribution,
                "data_gathering_status": state["gathered_data"].get("status"),
                "scoring_status": state["scored_providers"].get("status"),
                "validation_status": state["validation_results"].get("status"),
                "overall_confidence": self._calculate_overall_confidence(state),
                "execution_steps": len(state["execution_log"]),
                "errors_encountered": len(state["error_messages"]),
                "refinement": refinement_summary,
                # Drives the Responsible-AI panel's count (patient-facing, no
                # names) and the Agent Decision Process detail (developer, with
                # names). Both read this one structure so they cannot disagree.
                "withheld": shortfall,
                "cost_summary": get_cost_tracker().summary()
            }

            state["workflow_summary"] = workflow_summary

            self._log_step(state, "finalize_results", "completed", workflow_summary)

            return state

        except Exception as e:
            state["error_messages"].append(f"Result finalization failed: {str(e)}")
            self._log_step(state, "finalize_results", "failed", {"error": str(e)})
            return state

    def _handle_error(self, state: WorkflowState) -> WorkflowState:
        """Handle workflow errors and provide meaningful error responses."""
        try:
            self._log_step(state, "handle_error", "started", {
                "error_count": len(state["error_messages"])
            })

            # Create error summary
            error_summary = {
                "workflow_failed": True,
                "error_messages": state["error_messages"],
                "failed_step": state.get("current_step", "unknown"),
                "partial_results": {
                    "data_gathered": bool(state.get("gathered_data")),
                    "providers_scored": bool(state.get("scored_providers")),
                    "validation_completed": bool(state.get("validation_results"))
                }
            }

            state["workflow_summary"] = error_summary

            self._log_step(state, "handle_error", "completed", error_summary)

            return state

        except Exception as e:
            logger.error(f"Error handling failed: {e}")
            return state

    def _calculate_overall_confidence(self, state: WorkflowState) -> str:
        """Calculate overall confidence in the workflow results."""
        try:
            confidence_factors = []

            # Data gathering confidence
            if state["gathered_data"].get("status") == "success":
                provider_count = len(state["gathered_data"].get("providers", []))
                if provider_count >= 5:
                    confidence_factors.append("high")
                elif provider_count >= 2:
                    confidence_factors.append("medium")
                else:
                    confidence_factors.append("low")
            else:
                confidence_factors.append("low")

            # Scoring confidence
            if state["scored_providers"].get("status") == "success":
                confidence_factors.append("high")
            else:
                confidence_factors.append("low")

            # Validation confidence
            validation_confidence = state["validation_results"].get("validation_metadata", {}).get("ranking_confidence", "medium")
            confidence_factors.append(validation_confidence)

            # Calculate overall
            high_count = confidence_factors.count("high")
            medium_count = confidence_factors.count("medium")
            low_count = confidence_factors.count("low")

            if high_count >= 2 and low_count == 0:
                return "high"
            elif low_count >= 2:
                return "low"
            else:
                return "medium"

        except Exception:
            return "low"

    # Conditional edge functions
    def _check_data_gathering_success(self, state: WorkflowState) -> str:
        """Check if data gathering was successful."""
        if state.get("gathered_data", {}).get("status") == "success":
            return "success"
        return "error"

    def _check_scoring_success(self, state: WorkflowState) -> str:
        """Check if provider scoring was successful."""
        if state.get("scored_providers", {}).get("status") == "success":
            return "success"
        return "error"

    def _check_validation_success(self, state: WorkflowState) -> str:
        """Check if ranking validation was successful."""
        if state.get("validation_results", {}).get("status") == "success":
            return "success"
        return "error"

    def execute_workflow(self, specialty: str, location: str, insurance: Optional[str] = None, preferences: Optional[Dict[str, Any]] = None, use_cache: bool = True) -> Dict[str, Any]:
        """Execute the complete provider matching workflow.

        Args:
            specialty: Medical specialty to search for
            location: Location to search in
            insurance: Insurance type filter (optional)
            preferences: User preference weights (optional)

        Returns:
            Dictionary containing workflow results and recommendations
        """
        try:
            logger.info(f"Starting provider matching workflow: {specialty} in {location}")

            # Create initial state
            initial_state = WorkflowState(
                specialty=specialty,
                location=location,
                insurance=insurance,
                preferences=preferences or {},
                use_cache=use_cache,
                gathered_data={},
                scored_providers={},
                validation_results={},
                current_step="",
                workflow_id="",
                error_messages=[],
                execution_log=[],
                final_recommendations=[],
                workflow_summary={}
            )

            # Execute workflow
            result = self.workflow.invoke(initial_state)

            logger.info(f"Workflow completed with status: {result.get('workflow_summary', {}).get('overall_confidence', 'unknown')}")

            validation_results = result.get("validation_results", {})

            return {
                "success": len(result.get("error_messages", [])) == 0,
                "final_recommendations": result.get("final_recommendations", []),
                "workflow_summary": result.get("workflow_summary", {}),
                "cost_summary": result.get("workflow_summary", {}).get("cost_summary", {}),
                "execution_log": result.get("execution_log", []),
                "agent_outputs": {
                    "data_gatherer": result.get("gathered_data", {}),
                    "preference_scorer": result.get("scored_providers", {}),
                    "critic_validator": validation_results
                },
                "error_messages": result.get("error_messages", [])
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "final_recommendations": [],
                "workflow_summary": {"workflow_failed": True, "error": str(e)},
                "cost_summary": {},
                "execution_log": [],
                "agent_outputs": {},
                "error_messages": [str(e)]
            }

    def execute_workflow_streaming(
        self,
        specialty: str,
        location: str,
        insurance: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[ProgressCallback] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Execute workflow with streaming progress updates.

        Uses workflow.stream() instead of invoke() to enable real-time progress tracking.

        Args:
            specialty: Medical specialty to search for
            location: Location to search in
            insurance: Insurance type filter (optional)
            preferences: User preference weights (optional)
            progress_callback: Callback function for receiving progress updates

        Returns:
            Dictionary containing workflow results and recommendations (same format as execute_workflow)
        """
        self.progress_callback = progress_callback

        try:
            logger.info(f"Starting streaming workflow: {specialty} in {location}")

            # Create initial state (same as execute_workflow)
            initial_state = WorkflowState(
                specialty=specialty,
                location=location,
                insurance=insurance,
                preferences=preferences or {},
                use_cache=use_cache,
                gathered_data={},
                scored_providers={},
                validation_results={},
                current_step="",
                workflow_id="",
                error_messages=[],
                execution_log=[],
                final_recommendations=[],
                workflow_summary={}
            )

            # Execute workflow with streaming
            final_state = None
            for state_chunk in self.workflow.stream(initial_state):
                for step_name, step_state in state_chunk.items():
                    if step_name not in ["__start__", "__end__"]:
                        final_state = step_state

            if final_state is None:
                raise RuntimeError("Workflow stream did not produce final state")

            # Extract results (same format as execute_workflow)
            result = final_state
            validation_results = result.get("validation_results", {})

            logger.info(f"Streaming workflow completed with status: {result.get('workflow_summary', {}).get('overall_confidence', 'unknown')}")

            return {
                "success": len(result.get("error_messages", [])) == 0,
                "final_recommendations": result.get("final_recommendations", []),
                "workflow_summary": result.get("workflow_summary", {}),
                "cost_summary": result.get("workflow_summary", {}).get("cost_summary", {}),
                "execution_log": result.get("execution_log", []),
                "agent_outputs": {
                    "data_gatherer": result.get("gathered_data", {}),
                    "preference_scorer": result.get("scored_providers", {}),
                    "critic_validator": validation_results
                },
                "error_messages": result.get("error_messages", [])
            }

        except Exception as e:
            logger.error(f"Streaming workflow execution failed: {e}")
            return {
                "success": False,
                "final_recommendations": [],
                "workflow_summary": {"workflow_failed": True, "error": str(e)},
                "cost_summary": {},
                "execution_log": [],
                "agent_outputs": {},
                "error_messages": [str(e)]
            }
        finally:
            self.progress_callback = None


def create_orchestrator() -> ProviderMatchingOrchestrator:
    """Factory function to create a ProviderMatchingOrchestrator instance.

    Returns:
        ProviderMatchingOrchestrator instance
    """
    return ProviderMatchingOrchestrator()