# regulatory-intelligence-agent

A small **LangChain agent** that monitors **regulatory and filing intelligence** for a
company, sector, or topic, using **Nimble** as its web-retrieval layer.

Give it a subject. It runs several targeted web searches — SEC filings, enforcement
actions, investigations, policy/rule changes — reads the primary documents, and returns a
structured brief of the material developments with citations and a materiality grade.

This is **Pattern A**: the LangChain agent owns the workflow (what to search, how to read
it, how to synthesise), and Nimble is the tool it calls for web data.

## How it works

```
run.py ──> agent.py: create_agent(model=gpt-5.1, tools=[nimble_search], response_format=RegulatoryBrief)
                          │
                          └─ nimble_search  →  Nimble Search API (search_depth="standard",
                                                full_content=True for documents it will cite,
                                                include_domains scoped per pass)
```

- `config.py` — the agent's role (`SKILL`), objectives (`GOALS`), and source tiers
  (`SOURCE_TIERS`), plus the system prompt builder.
- `schema.py` — the `RegulatoryBrief` / `Development` Pydantic models used as the agent's
  structured output.
- `agent.py` — builds the LangChain agent and the `nimble_search` tool.
- `run.py` — CLI entrypoint.

### A note on the Nimble client

The `nimble_search` tool calls Nimble's official **`nimble-python`** SDK directly. That is
the same client the `langchain-nimble` package wraps internally; we call it directly
because, as of `langchain-nimble` 4.0.0, its search wrapper does not expose `full_content`
and its domain-scoped ranking is unreliable for primary-document retrieval (an
`include_domains=["sec.gov"]` query returns `sec.gov/ombuds` rather than the filing you
asked for). Nimble's **Agent API V2** is a better fit for `langchain-nimble` — see the
`company-due-diligence-agent` repo for that pattern.

## Setup

```bash
uv sync                       # or: pip install -e .
cp .env.example .env          # add NIMBLE_API_KEY and OPENAI_API_KEY
```

## Run

```bash
uv run python run.py "NVIDIA"
uv run python run.py "semiconductor export controls" --json brief.json
uv run python run.py "Microsoft" --model gpt-4o
```

Environment overrides: `OPENAI_MODEL` (default `gpt-5.1`), `NIMBLE_CONTENT_CHAR_CAP`
(default `6000`), `AGENT_RECURSION_LIMIT` (default `40`).

## Example

`examples/nvidia_brief.json` is a real run for `"NVIDIA"` — seven developments spanning an
8-K on the Hugging Face acquisition, off-balance-sheet data-center guarantees, a BIS
export-control rule, Section 232/301 trade measures, a debt shelf takedown, and an SEC
no-action letter, each citing the primary document on `sec.gov` or `federalregister.gov`.
