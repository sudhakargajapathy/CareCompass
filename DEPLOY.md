# 🚀 Deploying CareCompass to Hugging Face Spaces

This guide covers everything needed to get CareCompass running as a **Docker Space**
on Hugging Face.

---

## 1. The big picture

```
   ┌──────────────────────┐        git push        ┌──────────────────────────┐
   │  GitHub              │  ───────────────────>  │  Hugging Face Space      │
   │  sudhakargajapathy/  │   (or push both ways)  │  <user>/CareCompass      │
   │  CareCompass         │                        │  sdk: docker             │
   └──────────────────────┘                        └────────────┬─────────────┘
                                                                │
                          ┌─────────────────────────────────────┘
                          │  HF reads README.md frontmatter → builds Dockerfile
                          v
   ┌───────────────────────────────────────────────────────────────────────┐
   │  Space container  (runs as UID 1000, NOT root)                        │
   │                                                                       │
   │   ┌─────────────────────────────────────────────────────────────┐     │
   │   │  entrypoint.sh → streamlit run app.py --server.port=7860     │     │
   │   └──────────────────────────────┬──────────────────────────────┘     │
   │                                  │                                    │
   │      /home/user/app  ────────────┼─── chroma_db/   (ephemeral)        │
   │      (owned by UID 1000)         │    logs/        (ephemeral)        │
   │                                  │                                    │
   │      /data  ─────────────────────┴─── only if persistent storage      │
   │                                       is purchased (see §7)           │
   └───────────────────────────┬───────────────────────────────────────────┘
                               │  port 7860 (app_port in frontmatter)
                               v
                    https://<user>-carecompass.hf.space
```

---

## 2. What was already correct

```
  [x] README.md YAML frontmatter with sdk: docker + app_port: 7860
  [x] Dockerfile exists and exposes 7860
  [x] entrypoint.sh binds 0.0.0.0 (not localhost)
  [x] Secrets read at runtime via os.getenv() — no build-time secrets needed
  [x] .gitignore excludes .env, chroma_db/, logs/
```

## 3. What had to change (already applied in this branch)

```
  ┌────────────────────────┬──────────────────────────────────────────────────┐
  │ Problem                │ Fix                                              │
  ├────────────────────────┼──────────────────────────────────────────────────┤
  │ Container ran as root; │ Dockerfile now does:                             │
  │ HF forces UID 1000 →   │   RUN useradd -m -u 1000 user                    │
  │ ChromaDB could not     │   USER user                                      │
  │ create chroma.sqlite3, │   WORKDIR /home/user/app                         │
  │ which raises inside    │   COPY --chown=user . .                          │
  │ get_vector_store() and │ Every writable path is now owned by UID 1000.    │
  │ kills EVERY search     │                                                  │
  ├────────────────────────┼──────────────────────────────────────────────────┤
  │ pytest/pytest-cov etc. │ Split into requirements-dev.txt.                 │
  │ built into the image   │ Runtime image installs only what it needs.       │
  ├────────────────────────┼──────────────────────────────────────────────────┤
  │ folium, streamlit-     │ Removed — zero imports anywhere in the codebase.  │
  │ folium, beautifulsoup4 │ Shorter, more reliable Space builds.             │
  ├────────────────────────┼──────────────────────────────────────────────────┤
  │ No .dockerignore →     │ Added. Excludes .git, tests/, caches, local       │
  │ .git + tests uploaded  │ chroma_db/logs.                                  │
  ├────────────────────────┼──────────────────────────────────────────────────┤
  │ Streamlit telemetry /  │ STREAMLIT_SERVER_HEADLESS=true and               │
  │ interactive prompt     │ STREAMLIT_BROWSER_GATHER_USAGE_STATS=false       │
  │ on first boot          │ set as ENV in the Dockerfile.                    │
  ├────────────────────────┼──────────────────────────────────────────────────┤
  │ State lost on restart  │ entrypoint.sh auto-uses /data for ChromaDB +     │
  │ even with paid disk    │ audit logs when persistent storage is mounted.   │
  └────────────────────────┴──────────────────────────────────────────────────┘
```

---

## 4. ⚠️ Read this before you set `ENV`

CareCompass gates the UI behind `DatabaseAuthenticator`, which needs **PostgreSQL**.
A Space has no database, so the value of the `ENV` variable decides whether your
Space is usable at all:

```
                        ┌──────────────────────────────┐
                        │  Is AUTH_DATABASE_URL set?   │
                        └───────────┬──────────────────┘
                       no           │            yes
             ┌──────────────────────┴──────────────────────┐
             v                                             v
   ┌─────────────────────┐                    ┌──────────────────────────┐
   │  What is ENV?       │                    │  Real login form.        │
   └────┬───────────┬────┘                    │  Bootstraps admin from   │
        │           │                         │  APP_ADMIN_USERNAME /    │
 unset / │           │ production             │  APP_ADMIN_PASSWORD.     │
 development         │                        │        ✅ WORKS          │
        v            v                        └──────────────────────────┘
 ┌──────────────┐  ┌──────────────────────────┐
 │ Dev mode.    │  │ "Authentication database │
 │ Auto-login   │  │  is not configured."     │
 │ as dev_admin │  │                          │
 │              │  │   ❌ SPACE IS BRICKED    │
 │ ✅ WORKS —   │  │   Nobody can log in.     │
 │ open to      │  └──────────────────────────┘
 │ anyone with  │
 │ the URL      │
 └──────────────┘
```

**Recommendation for a public portfolio demo:** leave `ENV` **unset** (or
`development`) and leave `AUTH_DATABASE_URL` unset. Anyone with the URL can run
searches against *your* API keys — so pair this with a **private Space** or
tight `RATE_LIMIT_MAX_REQUESTS` if cost is a concern.

**Do not** set `ENV=production` unless you also supply a reachable
`AUTH_DATABASE_URL`.

---

## 5. Step-by-step deployment

### Step 1 — Create the Space

```
  huggingface.co  →  [ + New ]  →  Space
  ┌──────────────────────────────────────────────────┐
  │  Owner            : sudhakar1109               │
  │  Space name       : CareCompass                  │
  │  License          : MIT                          │
  │  Space SDK        : ● Docker   ○ Streamlit  ○ ... │  <-- MUST be Docker
  │  Docker template  : Blank                        │
  │  Hardware         : CPU basic (free) is enough    │
  │  Visibility       : Public / Private              │
  └──────────────────────────────────────────────────┘
                          [ Create Space ]
```

> Pick **Docker**, not Streamlit. The Streamlit SDK ignores the `Dockerfile`
> and would not give you the UID-1000-safe layout.

### Step 2 — Add secrets

`Space → Settings → Variables and secrets`

```
  ┌─── Secrets (encrypted, injected as env vars at runtime) ────────────┐
  │  OPENAI_API_KEY         sk-...                                      │
  │  APP_ANTHROPIC_API_KEY  sk-ant-...                                  │
  │  TAVILY_API_KEY         tvly-...                                    │
  │  ENCRYPTION_KEY         <fernet key, see below>          (optional) │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─── Variables (plaintext, visible to anyone) ────────────────────────┐
  │  RATE_LIMIT_MAX_REQUESTS   10                            (optional) │
  │  MAX_PROVIDERS_PER_SEARCH  20                            (optional) │
  └─────────────────────────────────────────────────────────────────────┘
```

Generate the optional encryption key locally:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

> All three API keys are **required** — `check_api_keys()` blocks the UI and
> shows "🚨 Missing API Keys" if any is absent.

### Step 3 — Push the code

```
  local repo                                     Hugging Face
  ──────────                                     ────────────
      │                                                │
      │  git remote add space \                        │
      │    https://huggingface.co/spaces/<user>/CareCompass
      │                                                │
      │  git push space claude/hugging-face-space-deploy-nhqxpz:main
      │  ───────────────────────────────────────────>  │
      │                                                │
      │                                    ┌───────────v───────────┐
      │                                    │ Building  ▓▓▓▓▓░░░░░  │
      │                                    │ ~5-10 min first build │
      │                                    └───────────┬───────────┘
      │                                                v
      │                                    ┌───────────────────────┐
      │                                    │  Running   ● green    │
      │                                    └───────────────────────┘
```

```bash
# authenticate once (needs a WRITE token from hf.co/settings/tokens)
pip install -U huggingface_hub
hf auth login

# add the Space as a second remote and push
git remote add space https://huggingface.co/spaces/<your-hf-user>/CareCompass
git push space claude/hugging-face-space-deploy-nhqxpz:main
```

> HF Spaces serve the **`main`** branch, so the `branch:main` refspec matters.

### Step 4 — Watch the build

```
  Space page → "Logs" dropdown
  ┌──────────────────┬──────────────────────────────────────────────┐
  │ Build            │ docker build output — pip install, apt, etc.  │
  │ Container        │ runtime stdout/stderr — streamlit + tracebacks│
  └──────────────────┴──────────────────────────────────────────────┘
```

Healthy container logs end with:

```
  You can now view your Streamlit app in your browser.
  Network URL: http://0.0.0.0:7860
```

---

## 6. Troubleshooting

```
  ┌───────────────────────────────┬─────────────────────────────────────────┐
  │ Symptom                       │ Cause / fix                             │
  ├───────────────────────────────┼─────────────────────────────────────────┤
  │ Build ok, Space stuck on      │ App not on 7860, or bound to 127.0.0.1. │
  │ "Starting"                    │ Check app_port: 7860 in frontmatter.    │
  ├───────────────────────────────┼─────────────────────────────────────────┤
  │ PermissionError / "attempt to │ A path is root-owned. Everything the    │
  │ write a readonly database"    │ app writes must be under /home/user or  │
  │                               │ /data, created after `USER user`.       │
  ├───────────────────────────────┼─────────────────────────────────────────┤
  │ "🚨 Missing API Keys"         │ Secret missing or misnamed. Note the    │
  │                               │ Anthropic one is APP_ANTHROPIC_API_KEY, │
  │                               │ not ANTHROPIC_API_KEY.                  │
  ├───────────────────────────────┼─────────────────────────────────────────┤
  │ "Authentication database is   │ ENV=production with no AUTH_DATABASE_URL│
  │  not configured."             │ → unset ENV. See §4.                    │
  ├───────────────────────────────┼─────────────────────────────────────────┤
  │ Results vanish after restart  │ Ephemeral disk. Expected. See §7.       │
  ├───────────────────────────────┼─────────────────────────────────────────┤
  │ Build times out               │ chromadb pulls onnxruntime (~200 MB).   │
  │                               │ Retry; the layer cache makes #2 faster. │
  └───────────────────────────────┴─────────────────────────────────────────┘
```

---

## 7. Storage: what survives a restart

```
   Free tier                          With persistent storage ($5+/mo)
   ─────────                          ────────────────────────────────
   50 GB ephemeral disk               /data mounted, survives restarts
        │                                       │
        │ restart / sleep                       │ restart / sleep
        v                                       v
   ┌──────────────┐                     ┌──────────────┐
   │  WIPED       │                     │  PRESERVED   │
   │  chroma_db/  │                     │  /data/chroma_db/
   │  logs/       │                     │  /data/logs/ │
   └──────────────┘                     └──────────────┘
```

`scripts/entrypoint.sh` detects a writable `/data` and points
`CHROMA_PERSIST_DIRECTORY` and `AUDIT_LOG_PATH` at it automatically — no config
change needed when you upgrade. For a demo, the ephemeral tier is fine: the
vector store is a cache, and every search re-gathers live data.

---

## 8. Optional — auto-sync GitHub → Space

Add `.github/workflows/sync-to-hf.yml` and set an `HF_TOKEN` repo secret
(a **write** token from https://huggingface.co/settings/tokens):

```yaml
name: Sync to Hugging Face Space
on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Push to Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git push --force \
            https://user:$HF_TOKEN@huggingface.co/spaces/<your-hf-user>/CareCompass \
            HEAD:main
```

---

## 9. Post-deploy checklist

```
  [ ] Space status is green "Running"
  [ ] Container logs show the Streamlit banner on 0.0.0.0:7860
  [ ] UI loads without the "Missing API Keys" banner
  [ ] Demo search — Neurology / "Phoenix, AZ" — returns ranked providers
  [ ] "Agent Workflow" panel shows all three agents completing
  [ ] Live demo URL added to the top of README.md
```
