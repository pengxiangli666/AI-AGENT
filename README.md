# coding-agent：企业级 AI Coding Agent 复现

LangGraph 多角色循环（Planner → Coder → Reviewer → Executor → Deployer）+ MCP 工具服务（SSE/HTTP）+ Docker/Seccomp 沙箱 + Nginx FaaS 部署 + BM25/BGE-M3 混合检索。

```
                 ┌────────────────────── agent (LangGraph) ──────────────────────┐
 requirement ──▶ │ planner ─▶ coder ─▶ reviewer ─▶ executor ─▶ [人工审批] ─▶ deployer │
                 │              ▲  └─ 不通过 ─┘        │ 失败(带 stderr)                │
                 │              └─────── 重试(≤ max_attempts) ◀──┘                    │
                 │   PostgresSaver 每步落盘 · interrupt_before=["deployer"]            │
                 └──────────────┬────────────────────────────────────────────────┘
                                │ MCP (SSE / streamable HTTP)
                 ┌──────────────▼───────── mcp_server (FastAPI) ─────────────────┐
                 │ run_in_sandbox · terminal_start/logs/kill · browse_page · ui_check │
                 │ deploy_faas · undeploy_faas · search_docs   (装饰器自动生成 schema)   │
                 └───────┬──────────────────┬─────────────────────┬──────────────┘
                         │                  │                     │
              sandbox/runner.py     Playwright + DOM 精简    sandbox/faas.py ─▶ Nginx
        docker run --cap-drop ALL            (SSRF 过滤)        (internal 网络, 动态 conf)
        seccomp · read-only · no-net
```

## 目录

| 路径 | 内容 |
|---|---|
| `agent/` | `state.py` StateSchema；`nodes.py` 五个节点 + 条件边；`graph.py` 建图、PostgresSaver、回滚；`llm.py` 分层模型路由；`main.py` CLI |
| `mcp_server/` | `registry.py` 自动发现工具并生成 JSON Schema；`server.py` FastAPI + SSE + streamable HTTP；`tools/` 各工具 |
| `sandbox/` | `runner.py` 沙箱执行 + 异步 PID/日志流；`faas.py` 部署与 Nginx；`Dockerfile`；`seccomp.json` |
| `rag/` | `index.py` BM25+Faiss+RRF+重排；`eval.py` 消融评测；`ingest.py` 建索引 |
| `scripts/` | `gen_seccomp.py` 生成 seccomp；`build_sandbox.sh` 构建镜像并测启动耗时 |
| `tests/` | 单元 / 端到端 / Postgres / Docker 集成测试 |

## 快速开始（Linux / WSL2）

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && playwright install chromium
cp .env.example .env            # 填 OPENAI_API_KEY，然后: set -a; . ./.env; set +a

docker compose up -d            # postgres + nginx（nginx 挂在 internal 网络 agent-faas 上）
bash scripts/build_sandbox.sh   # 构建沙箱镜像 + 打印 10 次启动耗时

python -m mcp_server.server &   # MCP 服务，默认 :8765  (/sse, /http/mcp, /healthz, /tools)
python -m agent.main "写一个 FastAPI 服务，GET /health 返回 {\"ok\": true}，带测试" --thread demo1
# 到部署前会暂停，输入 a 批准 / r 拒绝 / b 回滚到某个 checkpoint
# 进程中途挂了：python -m agent.main --thread demo1 --resume
```

部署后的 URL 形如 `http://app-1a2b3c4d.localhost:8080`（`*.localhost` 浏览器和 curl 会自动解析到本机）。

RAG：`python -m rag.ingest ./docs_corpus` 建索引（首次会下载 BGE-M3 约 2GB），MCP 工具 `search_docs` 即可用。
没有 GPU/不想下模型时，`RAG_BACKEND=hashing` 走离线替身（仅用于冒烟，**不能用来报召回率**）。

## 测试

```bash
pytest -q                                                     # 无需 Docker/Postgres 的全部测试
TEST_POSTGRES_URI=postgresql://agent:agent@localhost:5432/agent pytest tests/test_postgres_checkpointer.py
bash scripts/build_sandbox.sh && pytest tests/test_docker_integration.py -v   # 真 Docker 加固验证
```

## 设计要点（面试可讲）

- **自我纠错循环**：Reviewer 不通过 或 Executor 退出码非 0 都回到 Coder，并把 reviewer 意见 / 真实 stderr 拼进 prompt；`max_attempts`（默认 3）兜底防死循环，超限进入 `give_up`。Coder 产出的文件先过 `validate_files`（路径穿越、数量、大小），不合格直接回炉，不浪费 Reviewer 调用。
- **断点续跑**：`AsyncPostgresSaver` 每个超步写 checkpoint。Executor 崩溃后新进程用同一 `thread_id` 恢复，只重跑 Executor，不会重新规划/生成（`tests/test_postgres_checkpointer.py` 里用「没有任何回复的 LLM」证明）。
- **HITL**：`interrupt_before=["deployer"]`；Deployer 自身还会检查 `approved_by_human is True`，即使被误恢复也不会部署。回滚 = 从历史 checkpoint `aupdate_state` 分叉。
- **模型分层**：`ROLE_TIER` 显式表（planner/reviewer→便宜模型，coder→强模型）。简历里写的 `RunnableWithMessageHistory` 管的是会话历史，不是路由，这里没用它；`ModelRouter.usage` 按层统计真实 token。
- **MCP**：`@mcp_tool` + `autodiscover`：新增工具 = 在 `tools/` 加一个文件。docstring 的 `Args:` 被解析成 schema 的 description，并保证与 FastMCP 实际下发的 schema 一致（有测试）。同时暴露 SSE 与 streamable HTTP。
- **终端加固**：容器 `--cap-drop ALL`、`no-new-privileges`、自定义 seccomp、只读根文件系统、`--network none`、非 root、pids/内存/CPU 限额；超时用进程组 + `docker kill` 双保险。`terminal_start/logs/kill` 提供异步 PID 与增量日志。
- **seccomp 怎么来的**：`scripts/gen_seccomp.py` 以 Docker 默认 allowlist（默认拒绝）为底，删掉 `ptrace`/`process_vm_*`/`kcmp`/`pidfd_getfd` 等跨进程与遗留调用，socket 只留 UNIX/INET/INET6/NETLINK；`mount/unshare/setns/bpf` 等仍由 cap 门控，配合 `--cap-drop ALL` 被拒。
- **FaaS**：部署容器只在 `internal: true` 的 docker 网络里（无出网），Nginx 同时挂两张网；每个应用生成一个 server block，用 Docker DNS 变量式 `proxy_pass`，写入 → `nginx -t` → `reload`，任何一步失败回滚容器和配置。
- **浏览器**：DOM 精简（去 script/style/nav/footer/cookie 横幅等，保留标题/段落/代码/链接，限长）；URL 过滤挡住回环、内网、链路本地（云元数据 169.254.169.254）和非 http(s)，只放行自家 `FAAS_DOMAIN`。
- **混合检索**：BM25（标识符按 snake/Camel 拆分，中文用字符 bigram）+ BGE-M3 稠密向量（Faiss 精确内积），RRF（k=60）融合，再用 bge-reranker 重排前 30。`rag.eval` 一次输出 bm25 / dense / hybrid / hybrid+rerank 四档的 Recall@k 与 MRR@k。

## 简历里的数字怎么测出来（别直接抄）

| 简历说法 | 现在能怎么真实测量 |
|---|---|
| 启动 5 分钟 → <10 秒 | `scripts/build_sandbox.sh`：冷/热构建耗时 + 10 次容器启动均值。「5 分钟」需要你自己造基线（每次现装依赖） |
| 召回 92% | 自己标 50–200 条 query→相关 chunk 的评测集，跑 `python -m rag.eval --queries ...`，说清 Recall@几、是 hit-rate 还是 strict |
| token 成本 -40% | 同一批任务跑两遍（全用强模型 vs 分层），比较 `router.usage` 的 token 与实际账单 |
| 工程开销 -30% / 工具接入 -70% | 没有可复现的度量方式就不要写数字，改写成可验证的事实（如「新增工具只需 1 个文件、约 N 行」） |

## 已验证 / 未验证

- 已在无 Docker 守护进程的环境里跑通：图的所有分支（含重试上限、回滚、审批）、真实 MCP 服务器 SSE + streamable HTTP 往返、真实代码执行与 stderr 回灌、Postgres 16 上的崩溃恢复、混合检索与评测逻辑、seccomp 配置属性、FaaS/Nginx 编排（用假 docker）。
- **未在本环境验证**（需要你的机器）：真实 Docker 加固（`tests/test_docker_integration.py` 会自动在有 Docker 时执行）、Playwright 真实浏览器、OpenAI 实际调用、BGE-M3 / reranker 实际效果、Nginx 真正 reload。
- Postgres 数据库必须是 UTF8 编码（官方 `postgres` 镜像默认是）；SQL_ASCII 会让 LangGraph 的 checkpoint id 变成 bytes 而丢失 checkpoint。
