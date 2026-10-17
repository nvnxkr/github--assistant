"""
agents/supervisor.py

LangChain-based multi-agent system.
A Supervisor agent routes queries to specialised sub-agents:
  - CodeRetrievalAgent   → general code Q&A
  - ArchitectureAgent    → system design & structure
  - SecurityAgent        → security reviews
  - BugAnalysisAgent     → bug and code smell detection
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from retrieval.retriever import ContextualRetriever


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent:
    """Abstract base for all specialised agents."""

    name: str = "base"
    description: str = "Generic agent"
    system_prompt: str = "You are a helpful assistant."

    def __init__(
        self,
        retriever: ContextualRetriever,
        llm: BaseLanguageModel,
    ) -> None:
        self.retriever = retriever
        self.llm = llm

    def run(self, query: str) -> dict[str, Any]:
        docs = self.retriever.retrieve(query, k=6)
        context = self._format_context(docs)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({})
        return {"answer": answer, "sources": docs, "agent": self.name}

    @staticmethod
    def _format_context(docs: list[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            fp = doc.metadata.get("file_path", "?")
            parts.append(f"[{i}] {fp}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Specialised agents
# ---------------------------------------------------------------------------

class CodeRetrievalAgent(BaseAgent):
    name = "code_retrieval"
    description = "Answers general questions about code: finds functions, classes, explains logic."
    system_prompt = """You are an expert software engineer.
Your job is to answer questions about the codebase using the retrieved source code.
- Always cite file names and line numbers.
- Explain code in plain English.
- Show relevant code snippets when helpful."""


class ArchitectureAgent(BaseAgent):
    name = "architecture"
    description = "Analyses system design, folder structure, modules, and dependencies."
    system_prompt = """You are a software architect.
Analyse the repository structure and explain:
- High-level architecture (frontend / backend / database layers)
- Module responsibilities
- Key dependencies and how they interact
- Data flow through the system
Be concise and use diagrams in ASCII art when helpful."""


class SecurityAgent(BaseAgent):
    name = "security"
    description = "Reviews code for security vulnerabilities, auth issues, and secrets exposure."
    system_prompt = """You are a security expert performing a code security review.
Check for:
- Hardcoded secrets or API keys
- SQL injection vulnerabilities
- Insecure authentication (JWT issues, weak hashing)
- Missing input validation
- Insecure direct object references
- Exposed sensitive endpoints
Report each finding with: severity (HIGH/MEDIUM/LOW), location, and recommendation."""


class BugAnalysisAgent(BaseAgent):
    name = "bug_analysis"
    description = "Detects bugs, missing error handling, code smells, and anti-patterns."
    system_prompt = """You are a senior code reviewer.
Identify:
- Potential runtime errors and unhandled exceptions
- Logic bugs and off-by-one errors
- Missing null/None checks
- Infinite loops or recursion risks
- Code smells: duplicated logic, dead code, long functions
- Resource leaks (unclosed files/connections)
For each issue, provide: location, description, severity, and suggested fix."""


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """You are a routing supervisor for a GitHub repository Q&A system.
Given a user query, decide which specialist agent should handle it.

Agents available:
- code_retrieval: General code questions, function lookups, class explanations
- architecture: System design, folder structure, module relationships, dependencies
- security: Security review, vulnerability detection, auth analysis
- bug_analysis: Bug detection, error handling, code smells, refactoring

Respond with ONLY a JSON object:
{"agent": "<agent_name>", "reasoning": "<one sentence why>"}
"""


class Supervisor:
    """
    Routes user queries to the appropriate specialised agent.
    Uses LangChain prompt chains for routing and responses.
    """

    _AGENTS = {
        "code_retrieval": CodeRetrievalAgent,
        "architecture": ArchitectureAgent,
        "security": SecurityAgent,
        "bug_analysis": BugAnalysisAgent,
    }

    # Keyword-based fallback routing (no LLM required)
    _KEYWORD_MAP: dict[str, list[str]] = {
        "security": ["security", "vulnerability", "auth", "jwt", "password", "injection", "secret", "token", "exploit"],
        "bug_analysis": ["bug", "error", "crash", "smell", "refactor", "duplicate", "dead code", "improve"],
        "architecture": ["architecture", "structure", "design", "overview", "dependency", "module", "layer", "diagram"],
    }

    def __init__(
        self,
        retriever: ContextualRetriever,
        llm: BaseLanguageModel,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self._agents = {
            name: cls(retriever, llm)
            for name, cls in self._AGENTS.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> dict[str, Any]:
        """
        Route the query to the best agent and return the result.

        Returns:
            {
                "answer": str,
                "agent": str,
                "sources": list[Document],
                "routing_reason": str,
            }
        """
        agent_name, reason = self._route(query)
        logger.info(f"Routing to [{agent_name}]: {reason}")

        agent = self._agents[agent_name]
        result = agent.run(query)
        result["routing_reason"] = reason
        return result

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(self, query: str) -> tuple[str, str]:
        """Determine which agent should handle *query*."""
        # Try LLM routing first
        try:
            return self._llm_route(query)
        except Exception as exc:
            logger.warning(f"LLM routing failed ({exc}), using keyword fallback")
            return self._keyword_route(query)

    def _llm_route(self, query: str) -> tuple[str, str]:
        """Use the LLM to pick an agent."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=f"User query: {query}"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke({})

        # Strip markdown fences if present
        raw = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)
        agent_name = parsed.get("agent", "code_retrieval")
        reason = parsed.get("reasoning", "LLM routing decision")

        if agent_name not in self._agents:
            agent_name = "code_retrieval"

        return agent_name, reason

    def _keyword_route(self, query: str) -> tuple[str, str]:
        """Fallback: keyword-based routing."""
        q_lower = query.lower()
        for agent_name, keywords in self._KEYWORD_MAP.items():
            if any(kw in q_lower for kw in keywords):
                return agent_name, f"Keyword match for '{agent_name}'"
        return "code_retrieval", "Default general code retrieval"
