# coding-agent

An AI coding agent that plans, writes, reviews, runs and (optionally) deploys code.

A LangGraph state machine drives four roles (Planner, Coder, Reviewer, Executor). Generated code runs in a hardened Docker sandbox. HTTP services can be deployed behind Nginx after a human approves. Tools are exposed over MCP (SSE and streamable HTTP), and a hybrid retrieval stack (BM25 + BGE-M3 + reranker) serves documentation search.

```
                 ┌─────────────────────── agent (LangGraph) ───────────────────────┐
 requirement ──▶ │ planner ─▶ coder ─▶ reviewer ─▶ executor ─┬─▶ [human approval] ─▶ deployer │
                 │             ▲  └─ rejected ─┘     │        └─▶ finish (library / CLI)      │
                 │             └──── retry (≤ max_attempts, stderr fed back) ◀─┘              │
                 │      PostgresSaver checkpoint every step · interrupt_before=["deployer"]   │
                 └───────────────┬─────────────────────────────────────────────────┘
                                 │ MCP (SSE / streamable HTTP)
                 ┌───────────────▼────────── mcp_server (FastAPI) ─────────────────┐
                 │ run_in_sandbox · terminal_start/logs/kill · browse_page · ui_check │
                 │ deploy_faas · list_deployments · undeploy_faas · search_docs        │
                 │ (decorator + docstring → JSON Schema, auto-discovered)             │
                 └──────┬──────────────────┬──────────────────────┬───────────────┘
                        │                  │                      │
              sandbox/runner.py     Playwright + DOM trimming   sandbox/faas.py ─▶ Nginx
        docker run --cap-drop ALL          (SSRF guard)          (internal network, dynamic conf)
        seccomp · read-only · no network
```

## What it can do

| Task type | Behavior |
|---|---|
| Libraries, algorithms, utilities | Writes code and tests, runs them in the sandbox, retries on failure. Ends as `completed`, no approval or deploy. |
| CLI tools and scripts | Same as above. |
| HTTP web services (FastAPI, Flask, …) | Pauses for approval, then deploys to `http://app-<id>.localhost:8080`. |

The Planner decides with a `deployable` flag whether the result is a long-running HTTP service. If the flag is missing, the task is treated as deployable.

Current limits:

- The sandbox has no network, so only the standard library and packages baked into the image are available.
- Deployment start-up detection supports Python apps only.
- One container per deployment. No database or multi-service apps.
- The agent graph itself calls `run_in_sandbox` and `deploy_faas`. The retrieval and browser tools are available over MCP but are not yet part of the Coder's loop.

## Layout

| Path | Contents |
|---|---|
| `agent/` | `state.py` state schema · `nodes.py` planner / coder / reviewer / executor / deployer / finish + conditional edges · `graph.py` graph, PostgresSaver, rollback · `llm.py` tiered model routing · `main.py` CLI |
| `mcp_server/` | `registry.py` tool auto-discovery and schema generation · `server.py` FastAPI + SSE + streamable HTTP · `tools/` one file per tool group |
| `sandbox/` | `runner.py` sandboxed execution and async PID/log streaming · `faas.py` deployment and Nginx orchestration · `Dockerfile` · `seccomp.json` |
| `rag/` | `chunking.py` · `ingest.py` index build · `index.py` BM25 + Faiss + RRF + rerank · `relevance.py` gold matching · `eval.py` evaluation |
| `scripts/` | `build_sandbox.sh` · `gen_seccomp.py` · `gen_eval.py`, `refine_eval.py`, `rag_ablation.py` evaluation pipeline |
| `tests/` | Unit, end-to-end, Postgres and Docker integration tests |

## Quick start (Linux / WSL2)

Requirements: Python 3.11+, Docker, an OpenAI API key.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && playwright install chromium
cp .env.example .env              # fill in OPENAI_API_KEY (keep .env out of git)

docker compose up -d              # Postgres (host port 5433) + Nginx (host port 8080)
bash scripts/build_sandbox.sh     # build the sandbox image, print 10 start-up timings

python -m mcp_server.server &     # MCP server on :8765  (/sse, /http/mcp, /healthz, /tools)
python -m agent.main "Write a FastAPI service with GET /health returning {\"ok\": true}, with tests" --thread demo1
```

CLI options:

| Option | Meaning |
|---|---|
| `--thread ID` | Name of the run. Reuse it to resume. |
| `--max-attempts N` | Retry cap for the Coder loop (default 3). |
| `--out DIR` | Write the generated files to `DIR` when the run ends. |
| `--resume` | Continue a thread after a crash: `python -m agent.main --thread demo1 --resume` |

At the approval step: `a` approves, `r` rejects, `b` rolls back to an earlier checkpoint.

Keep the MCP server bound to `127.0.0.1` (the default). It can run code and deploy containers and has no authentication.

## Configuration

All settings live in `.env` (see `.env.example`).

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | – | Credentials and endpoint, shared by both tiers |
| `MODEL_ROUTER` | `gpt-4o-mini` | Cheap tier: Planner, Reviewer, Deployer |
| `MODEL_GENERATOR` | `gpt-4o` | Strong tier: Coder |
| `POSTGRES_URI` | `…@localhost:5433/agent` | Checkpoint store (database must be UTF8) |
| `MCP_HOST`, `MCP_PORT`, `MCP_URL` | `127.0.0.1`, `8765` | MCP server |
| `SANDBOX_IMAGE`, `SANDBOX_TIMEOUT` | `coding-agent-sandbox:latest`, `60` | Sandbox image and run timeout (s) |
| `FAAS_DOMAIN`, `FAAS_PUBLIC_PORT` | `localhost`, `8080` | Deployed-app hostnames |
| `NGINX_CONF_DIR` | `./deploy/nginx/conf.d` | Where per-app Nginx configs are written |
| `EMBED_MODEL`, `RERANK_MODEL`, `RAG_INDEX_DIR` | BGE-M3, bge-reranker-v2-m3, `./data/index` | Retrieval models and index location |
| `RAG_BACKEND` | – | `hashing` selects an offline stand-in for smoke tests only |
| `HF_TOKEN` | – | Optional. Avoids Hugging Face rate limits on model download |

## Design notes

**Self-correcting loop.** A Reviewer rejection or a non-zero Executor exit both return to the Coder, with the reviewer feedback or the real stderr added to the prompt. `max_attempts` bounds the loop and ends in `give_up`, which shows the last sandbox output. Generated files pass `validate_files` first (path traversal, file count, size), so invalid output does not cost a Reviewer call.

**Resume after a crash.** `AsyncPostgresSaver` writes a checkpoint at every step. A new process with the same `thread_id` continues from the last one; after an Executor crash only the Executor reruns, with no new planning or generation. `tests/test_postgres_checkpointer.py` proves this with an LLM that has no replies queued.

**Human approval.** The graph stops before `deployer` (`interrupt_before`). The Deployer also checks `approved_by_human is True`, so a mistaken resume still cannot deploy. Rollback forks from a past checkpoint with `aupdate_state`.

**Tiered models.** An explicit `ROLE_TIER` table maps planner, reviewer and deployer to the cheap tier and coder to the strong tier. `ModelRouter.usage` records real token counts per tier. Both tiers use the OpenAI-compatible endpoint.

**MCP tools.** `@mcp_tool` plus `autodiscover` means adding a tool is adding one file under `tools/`. The `Args:` section of the docstring becomes parameter descriptions in the JSON Schema, and a test checks that it matches the schema FastMCP publishes. Both SSE and streamable HTTP are served.

**Sandbox hardening.** Each run uses `--network none`, `--cap-drop ALL`, `no-new-privileges`, a custom seccomp profile, a read-only root filesystem with tmpfs scratch space, a non-root user (10001), and pids, memory and CPU limits. Timeouts kill the process group and the container. `terminal_start/logs/kill` give asynchronous PIDs and incremental logs by polling.

**Seccomp profile.** `scripts/gen_seccomp.py` starts from Docker's default allowlist (default deny), removes `ptrace`, `process_vm_*`, `kcmp`, `pidfd_getfd` and similar cross-process calls, and restricts sockets to UNIX, INET, INET6 and NETLINK. `mount`, `unshare`, `setns` and `bpf` stay blocked by the dropped capabilities.

**Deployment.** Deployed containers sit only on the `internal: true` network `agent-faas`, so they cannot reach the internet. Nginx is attached to both networks. Each app gets a server block that uses Docker DNS with a variable `proxy_pass`. The sequence is write config, `nginx -t`, reload; any failure rolls back the container and the config. A readiness probe waits for the app to answer HTTP before the URL is returned.

**Browser tools.** `browse_page` trims the DOM (scripts, styles, navigation, footers, cookie banners) and keeps headings, paragraphs, code and links, with a length cap. The SSRF guard resolves every host and refuses private, loopback, link-local (including the cloud metadata address), reserved and multicast addresses and non-HTTP(S) schemes. The only exception is a deployed app host `app-<8 hex>.<FAAS_DOMAIN>` on the public port. Ordinary public sites are allowed. `ui_check` runs assertions against a page and reports pass or fail.

## Retrieval

Pipeline: BM25 (identifiers split on snake_case and CamelCase, CJK bigrams) and BGE-M3 dense vectors (Faiss exact inner product) are fused with RRF (k=60). The top 30 fused candidates are reranked with bge-reranker-v2-m3.

Chunking: fixed-size by default. Heading-aware chunking (`rag/chunking.py`, `--split heading`) and header-prefixed chunks (`--header`) are available.

Build an index:

```bash
python -m rag.ingest ./docs_corpus            # first run downloads BGE-M3 (~2 GB)
python -m rag.ingest ./docs_corpus --split heading --header
```

### Evaluation

The evaluation set is synthetic. `scripts/gen_eval.py` asks an LLM to write one question per sampled chunk, and `scripts/refine_eval.py` audits them (dropping vague or multi-answer questions) and extracts verbatim evidence sentences. A retrieved chunk counts as relevant when it comes from the right file and contains at least 80% of the evidence's word 4-grams, so the labels do not depend on how the text was chunked. Questions are split 50/50 into dev (used for tuning) and test (reported once).

Corpus: LangGraph and pydantic documentation, 893 chunks. Test split: 196 questions. Recall@k is the fraction of questions whose relevant chunk appears in the top k; MRR@10 rewards earlier ranks.

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.515 | 0.801 | 0.842 | 0.626 |
| Dense (BGE-M3) | 0.602 | 0.806 | 0.862 | 0.692 |
| Hybrid (RRF) | 0.633 | 0.867 | 0.934 | 0.740 |
| Hybrid + rerank (top 30) | **0.724** | **0.934** | **0.974** | **0.819** |

The 95% Wilson interval for hybrid + rerank Recall@5 is [0.890, 0.961]. Heading chunking, header context and a larger rerank pool (50) gave no reliable gain, so the defaults stay as above. These numbers come from a synthetic question set and one corpus; treat them as a comparison between methods, not an absolute accuracy.

Reproduce:

```bash
python scripts/gen_eval.py            # writes data/eval.jsonl
python scripts/refine_eval.py         # writes data/eval_clean.jsonl (dev/test split)
python -m rag.eval --split test       # per-method table with confidence intervals
python scripts/rag_ablation.py        # chunking / header / pool-size ablation
```

## Tests

```bash
pytest -q                                                    # no Docker or Postgres needed
TEST_POSTGRES_URI=postgresql://agent:agent@localhost:5433/agent pytest tests/test_postgres_checkpointer.py
bash scripts/build_sandbox.sh && pytest tests/test_docker_integration.py -v    # real hardened containers
```

Coverage: every graph branch (retry cap, rollback, approval, library vs service), MCP round trips over SSE and streamable HTTP, real code execution with stderr feedback, crash recovery on Postgres 16, seccomp profile properties, FaaS and Nginx orchestration, browser DOM trimming and SSRF rules, retrieval and evaluation logic.

## Known gaps

- Terminal logs are polled, not streamed.
- The Coder does not yet call `search_docs`, and deployments are not yet checked automatically with `ui_check`.
- Python only for deployment; no network inside the sandbox.
- A deployed container once disappeared a few minutes after start while its Nginx config remained. The cause was not found and it did not recur.
- The Postgres database must use UTF8. With SQL_ASCII, LangGraph checkpoint ids come back as bytes and checkpoints are lost.
