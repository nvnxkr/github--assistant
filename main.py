"""Streamlit user interface for the GitHub RAG Assistant.

Run with:
    streamlit run main.py
"""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from app import GitHubRAGAssistant, RepositoryMetadata
from config import get_config
from ingestion.github_loader import validate_github_url


st.set_page_config(
    page_title="GitHub RAG Assistant",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_assistant(provider: str, cache_version: int = 2) -> GitHubRAGAssistant:
    """Keep one assistant instance per provider and Streamlit process."""
    config = get_config().model_copy(deep=True)
    config.llm.llm_provider = provider
    return GitHubRAGAssistant(config)


def metadata_summary(metadata: RepositoryMetadata) -> dict[str, Any]:
    return {
        "Files": metadata.file_count,
        "Chunks": metadata.chunk_count,
        "Size (MB)": round(metadata.total_size_mb, 2),
        "Languages": ", ".join(
            f"{language} ({count})"
            for language, count in sorted(metadata.language_breakdown.items())
        ) or "—",
    }


def render_query_error(exc: Exception) -> None:
    """Turn provider rate-limit errors into actionable UI guidance."""
    message = str(exc)
    if "429" in message or "rate limit" in message.lower() or "rate_limited" in message.lower():
        st.error(
            "The repository was indexed successfully, but the configured LLM "
            "provider rate-limited the answer request. Wait and try again, "
            "check your provider quota/API plan, or change `LLM_PROVIDER` in "
            "`.env` to `groq` or `local`."
        )
        with st.expander("Provider response"):
            st.code(message)
        return
    st.error(f"Unable to answer question: {message}")


def render_sources(result: dict[str, Any]) -> None:
    sources = result.get("sources", [])
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for source in sources:
                metadata = source.metadata if hasattr(source, "metadata") else source
                location = metadata.get("file_path", "Unknown file")
                start_line = metadata.get("start_line")
                end_line = metadata.get("end_line")
                if start_line and end_line:
                    location += f":{start_line}-{end_line}"
                st.code(location, language="text")

    citations = result.get("citations", "")
    if citations:
        st.markdown(citations)


def ingest_repository(assistant: GitHubRAGAssistant, url: str) -> None:
    if not validate_github_url(url):
        st.error("Enter a valid GitHub URL, such as `https://github.com/owner/repository`.")
        return

    with st.status("Indexing repository...", expanded=True) as status:
        try:
            st.write("Cloning repository and creating embeddings...")
            metadata = assistant.ingest_github_url(url)
            st.session_state["repository"] = metadata
            st.session_state["messages"] = []
            st.session_state["session_id"] = uuid.uuid4().hex
            status.update(label="Repository indexed", state="complete")
            st.success(
                f"Indexed {metadata.file_count} files and {metadata.chunk_count} chunks."
            )
        except Exception as exc:
            status.update(label="Indexing failed", state="error")
            st.error(f"Unable to index repository: {exc}")


def ingest_zip(assistant: GitHubRAGAssistant, uploaded_file: Any) -> None:
    with st.status("Indexing ZIP archive...", expanded=True) as status:
        try:
            st.write("Extracting files and creating embeddings...")
            metadata = assistant.ingest_zip(zip_bytes=uploaded_file.getvalue())
            st.session_state["repository"] = metadata
            st.session_state["messages"] = []
            st.session_state["session_id"] = uuid.uuid4().hex
            status.update(label="Archive indexed", state="complete")
            st.success(
                f"Indexed {metadata.file_count} files and {metadata.chunk_count} chunks."
            )
        except Exception as exc:
            status.update(label="Indexing failed", state="error")
            st.error(f"Unable to index archive: {exc}")


def render_sidebar(assistant: GitHubRAGAssistant, provider: str) -> None:
    with st.sidebar:
        st.header("Repository")
        url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repository")
        if st.button("Index GitHub repository", use_container_width=True):
            if not url.strip():
                st.error("Enter a GitHub URL first.")
            else:
                ingest_repository(assistant, url.strip())

        uploaded_file = st.file_uploader("Or upload a repository ZIP", type=["zip"])
        if st.button("Index ZIP archive", use_container_width=True):
            if uploaded_file is None:
                st.error("Choose a ZIP archive first.")
            else:
                ingest_zip(assistant, uploaded_file)

        st.divider()
        st.header("Session")
        use_agent = st.checkbox("Use supervisor agent", value=False)
        if st.button("Clear conversation", use_container_width=True):
            assistant.clear_history(st.session_state["session_id"])
            st.session_state["messages"] = []
            st.rerun()

        if st.button("Reset repository index", use_container_width=True):
            assistant.reset_index()
            st.session_state["repository"] = RepositoryMetadata()
            st.session_state["messages"] = []
            st.success("Repository index reset.")

        repository = st.session_state["repository"]
        if repository.file_count:
            st.divider()
            st.subheader("Indexed repository")
            st.json(metadata_summary(repository))

        with st.expander("Statistics"):
            try:
                st.json(assistant.get_stats())
            except Exception as exc:
                st.error(f"Unable to load statistics: {exc}")

        st.session_state["use_agent"] = use_agent
        st.caption(f"Active provider: `{provider}`")


def render_chat(assistant: GitHubRAGAssistant) -> None:
    st.title("GitHub RAG Assistant")
    st.info("Please add a GitHub URL in the sidebar to get started.")
    st.caption("Index a repository, then ask questions about its codebase.")

    if not assistant.has_indexed_repository():
        st.info("Index a GitHub repository or upload a ZIP archive from the sidebar before chatting.")

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("result", {}))

    question = st.chat_input(
        "Ask a question about the indexed repository...",
        disabled=not assistant.has_indexed_repository(),
    )
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the repository..."):
            try:
                result = assistant.ask(
                    question,
                    session_id=st.session_state["session_id"],
                    use_agent=st.session_state["use_agent"],
                )
            except Exception as exc:
                render_query_error(exc)
                return

        st.markdown(result.get("answer", "No answer was returned."))
        render_sources(result)
        st.caption(
            f"Agent: {result.get('agent', 'rag_chain')} · "
            f"Latency: {result.get('latency', 0):.3f}s"
        )

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": result.get("answer", "No answer was returned."),
            "result": result,
        }
    )


def main() -> None:
    configured_provider = get_config().llm.llm_provider
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "repository" not in st.session_state:
        st.session_state["repository"] = RepositoryMetadata()
    if "use_agent" not in st.session_state:
        st.session_state["use_agent"] = False
    if "provider" not in st.session_state:
        st.session_state["provider"] = configured_provider

    provider_labels = {
        "Groq": "groq",
        "Mistral": "mistral",
    }
    provider_options = list(provider_labels)
    current_provider = st.session_state["provider"]
    if current_provider not in provider_labels.values():
        current_provider = (
            configured_provider
            if configured_provider in provider_labels.values()
            else "mistral"
        )
    selected_label = st.sidebar.selectbox(
        "LLM provider",
        options=provider_options,
        index=[*provider_labels.values()].index(current_provider),
        help="Choose which configured chat model generates answers.",
    )
    provider = provider_labels[selected_label]
    if provider != current_provider:
        st.session_state["provider"] = provider
        st.session_state["messages"] = []
        st.session_state["session_id"] = uuid.uuid4().hex
    else:
        st.session_state["provider"] = provider

    assistant = get_assistant(provider)
    render_sidebar(assistant, provider)
    render_chat(assistant)


if __name__ == "__main__":
    main()
