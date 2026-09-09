"""Regulatory & filing intelligence agent.

A LangChain agent (Pattern A) that controls the research workflow and uses Nimble's
Search API as its web-retrieval tool. The agent decides what to search, reads the
primary documents, and synthesises a structured RegulatoryBrief.

Retrieval note: the tool calls Nimble's official Python SDK (``nimble-python``, the
same client ``langchain-nimble`` wraps) directly, because as of langchain-nimble
4.0.0 the search wrapper does not expose ``full_content`` and its domain-scoped
ranking is weak for primary-document retrieval. The Agent API examples (see the
company-due-diligence-agent repo) do use the ``langchain-nimble`` V2 tools.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from nimble_python import Nimble

from config import build_system_prompt
from schema import RegulatoryBrief

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")
DEFAULT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "40"))
# Primary filings can be > 300 KB; keep each result readable for the LLM.
CONTENT_CHAR_CAP = int(os.getenv("NIMBLE_CONTENT_CHAR_CAP", "6000"))


def _compact(results, cap: int) -> list[dict]:
    out = []
    for r in results or []:
        content = getattr(r, "content", "") or ""
        if len(content) > cap:
            content = content[:cap] + "\n...[truncated]"
        out.append(
            {
                "title": getattr(r, "title", None),
                "url": getattr(r, "url", None),
                "description": getattr(r, "description", None),
                "content": content,
            }
        )
    return out


def _make_search_tool():
    """Nimble Search API exposed as the agent's one retrieval tool."""
    client = Nimble()  # reads NIMBLE_API_KEY from the environment

    @tool
    def nimble_search(
        query: str,
        num_results: int = 8,
        read_full_pages: bool = False,
        include_domains: Optional[List[str]] = None,
        time_range: Optional[str] = None,
        start_date: Optional[str] = None,
    ) -> list[dict]:
        """Search the live web via Nimble. Returns [{title, url, description, content}].

        read_full_pages=False: fast scan, snippet-level content, use 8-10 results.
        read_full_pages=True:  pulls full page text for each hit - use for primary
                               documents you will cite, with num_results <= 5.
        include_domains: whitelist, e.g. ["sec.gov"] or ["justice.gov","ftc.gov"].
        time_range: one of hour/day/week/month/year. start_date: "YYYY-MM-DD".
        Pass either time_range or start_date, not both.
        """
        kwargs = {"query": query, "search_depth": "standard"}
        kwargs["max_results"] = min(num_results, 5) if read_full_pages else num_results
        if read_full_pages:
            kwargs["full_content"] = True
        if include_domains:
            kwargs["include_domains"] = include_domains
        if start_date:  # start_date and time_range are mutually exclusive in the API
            kwargs["start_date"] = start_date
        elif time_range:
            kwargs["time_range"] = time_range
        try:
            resp = client.search(**kwargs)
        except Exception as exc:  # let the agent see the error and retry differently
            return [{"error": f"{type(exc).__name__}: {exc}"}]
        cap = CONTENT_CHAR_CAP if read_full_pages else 1200
        return _compact(resp.results, cap)

    return nimble_search


def _make_llm(model: str | None):
    name = model or DEFAULT_MODEL
    # gpt-5.x / reasoning models only accept the default temperature.
    if name.startswith(("gpt-5", "o1", "o3", "o4")):
        return ChatOpenAI(model=name)
    return ChatOpenAI(model=name, temperature=0)


def build_agent(model: str | None = None, today: str | None = None):
    """Build the LangChain agent graph."""
    today = today or dt.date.today().isoformat()
    return create_agent(
        model=_make_llm(model),
        tools=[_make_search_tool()],
        system_prompt=build_system_prompt(today),
        response_format=RegulatoryBrief,
    )


def research(subject: str, model: str | None = None) -> RegulatoryBrief:
    """Run the agent end to end and return the structured brief."""
    today = dt.date.today().isoformat()
    agent = build_agent(model, today)
    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Research recent regulatory and filing developments related to "
                    f"{subject}. Identify material SEC disclosures, regulatory actions, "
                    f"investigations, or policy developments that could affect the "
                    f"subject. Explain what changed, why it matters, and cite the "
                    f"underlying evidence. Today is {today}.",
                )
            ]
        },
        {"recursion_limit": DEFAULT_RECURSION_LIMIT},
    )
    return result["structured_response"]
