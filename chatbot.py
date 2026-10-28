
"""Answer generation using Groq."""

import os
from typing import Sequence

from openai import OpenAI
from rag import RetrievedChunk, RAGError

NO_INFORMATION_MESSAGE = (
    "The information is not available in the provided volunteer information."
)

SYSTEM_PROMPT = """You are the AI Volunteer Chatbot.
Answer only using the supplied volunteer context.
Do not guess or use outside knowledge.
If the context does not contain the answer, reply exactly:
The information is not available in the provided volunteer information.
Keep answers clear and concise."""


def answer_question(
    client,
    model: str,
    question: str,
    retrieved_chunks: Sequence[RetrievedChunk],
) -> str:
    if not question.strip():
        raise ValueError("Please enter a question.")

    if not retrieved_chunks:
        return NO_INFORMATION_MESSAGE

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RAGError(
            "Groq API key is missing. Add GROQ_API_KEY to Streamlit Cloud Secrets."
        )

    context = "\n\n".join(
        f"[Source: {chunk.source}]\n{chunk.text}"
        for chunk in retrieved_chunks
    )

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0,
        )
    except Exception as exc:
        raise RAGError(
            "Groq could not generate an answer. Check your API key, "
            "model name, usage limits, and internet connection."
        ) from exc

    answer = (response.choices[0].message.content or "").strip()
    return answer or NO_INFORMATION_MESSAGE
