# syntax=docker/dockerfile:1.7
# Sandbox image for running model-generated code.
#
# Layer order is deliberate - slow, rarely-changing layers first so Docker's layer
# cache (and BuildKit cache mounts) make rebuilds fast, and runtimes + common
# packages are baked in so a run never has to install anything (the sandbox has no
# network).  Edit requirements-sandbox.txt / the npm line and only the later layers rebuild.
FROM python:3.12-alpine AS base

# 1. OS packages + Node runtime (changes rarely)
RUN apk add --no-cache nodejs npm bash coreutils tini

# 2. Python packages the generated projects can import offline
COPY sandbox/requirements-sandbox.txt /tmp/requirements-sandbox.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-sandbox.txt && rm /tmp/requirements-sandbox.txt

# 3. Node packages available globally (resolved through NODE_PATH)
RUN --mount=type=cache,target=/root/.npm \
    npm install -g --no-audit --no-fund express@4 jest@29 supertest@7
ENV NODE_PATH=/usr/local/lib/node_modules

# 4. Unprivileged user; /workspace is bind-mounted at run time
RUN addgroup -g 10001 sandbox && adduser -D -u 10001 -G sandbox -h /tmp sandbox
WORKDIR /workspace
USER 10001:10001

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
ENTRYPOINT ["/sbin/tini", "--"]
CMD ["sh"]
