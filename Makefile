.PHONY: test up sandbox serve

test:
	pytest -q

up:
	docker compose up -d

sandbox:
	bash scripts/build_sandbox.sh

serve:
	python -m mcp_server.server
