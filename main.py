

"""Streamlit entry point for the AI Volunteer Chatbot."""

import streamlit as st

from chatbot import NO_INFORMATION_MESSAGE, answer_question
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    KNOWLEDGE_DIR,
    RELEVANCE_THRESHOLD,
)
from rag import RAGError, RAGRetriever


st.set_page_config(page_title="AI Volunteer Chatbot", page_icon="🤝")
st.title("AI Volunteer Chatbot")
st.caption("Answers are based only on the organization's provided information.")


def get_retriever() -> RAGRetriever:
    """Build the index once per Streamlit session."""
    if "retriever" not in st.session_state:
        st.session_state.retriever = RAGRetriever.build(
            knowledge_dir=KNOWLEDGE_DIR,
            embedding_model_name=EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    return st.session_state.retriever


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + ", ".join(message["sources"]))

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

question = st.chat_input("Ask a question about volunteering...")

if question is not None:
    question = question.strip()

    if not question:
        st.warning("Please enter a question.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                retriever = get_retriever()
                matches = retriever.retrieve(
                    question, threshold=RELEVANCE_THRESHOLD
                )

                response = answer_question(
                    client=None,
                    model="openai/gpt-oss-120b",
                    question=question,
                    retrieved_chunks=matches,
                )

                sources = sorted({match.source for match in matches})
                if response == NO_INFORMATION_MESSAGE:
                    sources = []

                st.markdown(response)

                if sources:
                    st.caption("Sources: " + ", ".join(sources))

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "sources": sources,
                    }
                )

            except (RAGError, ValueError) as exc:
                st.error(str(exc))
