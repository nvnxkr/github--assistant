"""
chains/rag_chain.py

Main RAG pipeline built with LangChain LCEL.
Supports conversational memory and source citations.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from loguru import logger

from config import LLMConfig, get_config
from retrieval.retriever import ContextualRetriever


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert software engineer and code analyst.
You are helping a developer understand a GitHub repository they have loaded.

Use the retrieved code snippets below to answer the question.
Always cite specific file names and line numbers when referencing code.
If you are not certain about something, say so clearly.
Do NOT make up code or function names that are not in the context.

Retrieved context:
{context}

Source files used:
{sources}
"""

CONDENSE_QUESTION_PROMPT = """Given the following conversation history and a follow-up question,
rephrase the follow-up question to be a standalone question that captures all relevant context.

Chat History:
{chat_history}

Follow-up question: {question}

Standalone question:"""


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def build_llm(config: LLMConfig | None = None) -> BaseLanguageModel:
    """Build the configured LLM."""
    cfg = config or get_config().llm
    provider = cfg.llm_provider
    logger.info(f"Building LLM: provider={provider}")

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(
            model=cfg.mistral_model,
            mistral_api_key=cfg.mistral_api_key,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            max_retries=cfg.max_retries,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=cfg.groq_model,
            groq_api_key=cfg.groq_api_key,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            max_retries=cfg.max_retries,
        )
    elif provider == "local":
        from langchain_community.llms import Ollama
        return Ollama(
            base_url=cfg.local_llm_url,
            model=cfg.local_llm_model,
            temperature=cfg.temperature,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------

def _format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        fp = doc.metadata.get("file_path", "unknown")
        lang = doc.metadata.get("language", "")
        fn = doc.metadata.get("function_name", "")
        cl = doc.metadata.get("class_name", "")

        header_parts = [f"[{i}] File: {fp}"]
        if fn:
            header_parts.append(f"Function: {fn}")
        if cl:
            header_parts.append(f"Class: {cl}")
        header = " | ".join(header_parts)

        parts.append(f"{header}\n```{lang}\n{doc.page_content}\n```")
    return "\n\n---\n\n".join(parts)


def _format_sources(docs: list[Document]) -> str:
    """Format source citations."""
    lines = []
    for i, doc in enumerate(docs, 1):
        fp = doc.metadata.get("file_path", "unknown")
        start = doc.metadata.get("start_line", "?")
        end = doc.metadata.get("end_line", "?")
        fn = doc.metadata.get("function_name", "")
        lines.append(f"[{i}] {fp}" + (f"::{fn}" if fn else "") + f" (lines {start}–{end})")
    return "\n".join(lines)


class RAGChain:
    """
    Conversational RAG chain.

    Features:
    - LCEL-based pipeline
    - Conversational memory (chat history)
    - Source citations in every response
    - Streaming support
    """

    def __init__(
        self,
        retriever: ContextualRetriever,
        llm: BaseLanguageModel | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm or build_llm()
        self._chat_histories: dict[str, ChatMessageHistory] = {}
        self._chain = self._build_chain()
        self._chain_with_history = self._build_chain_with_history()

    # ------------------------------------------------------------------
    # Chain construction
    # ------------------------------------------------------------------

    def _build_chain(self):
        """Build the core LCEL RAG chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        return (
            RunnablePassthrough.assign(
                context=lambda x: _format_docs(
                    self.retriever.retrieve(x["question"])
                ),
                sources=lambda x: _format_sources(
                    self.retriever.retrieve(x["question"])
                ),
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _build_chain_with_history(self):
        """Wrap the chain with persistent message history."""
        return RunnableWithMessageHistory(
            self._chain,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self._chat_histories:
            self._chat_histories[session_id] = ChatMessageHistory()
        return self._chat_histories[session_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        session_id: str = "default",
    ) -> dict[str, Any]:
        """
        Answer a question using RAG with conversational memory.

        Returns:
            {
                "answer": str,
                "sources": list[Document],
                "citations": str,
            }
        """
        sources = self.retriever.retrieve(question)
        logger.info(f"RAG query [{session_id}]: {question[:80]}")

        answer = self._chain_with_history.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}},
        )

        return {
            "answer": answer,
            "sources": sources,
            "citations": _format_sources(sources),
        }

    def stream(
        self,
        question: str,
        session_id: str = "default",
    ) -> Iterator[str]:
        """Stream the answer token-by-token."""
        sources = self.retriever.retrieve(question)
        logger.info(f"RAG stream [{session_id}]: {question[:80]}")

        for chunk in self._chain_with_history.stream(
            {"question": question},
            config={"configurable": {"session_id": session_id}},
        ):
            yield chunk

    def clear_history(self, session_id: str = "default") -> None:
        """Clear chat history for a session."""
        if session_id in self._chat_histories:
            self._chat_histories[session_id].clear()
            logger.info(f"Cleared history for session: {session_id}")

    def get_history(self, session_id: str = "default") -> list[dict]:
        """Return chat history as a list of message dicts."""
        history = self._get_session_history(session_id)
        messages = []
        for msg in history.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        return messages
