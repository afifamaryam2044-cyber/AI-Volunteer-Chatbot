
"""Answer generation using local Ollama."""

from typing import Sequence
import ollama

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

    context = "\n\n".join(
        f"[Source: {chunk.source}]\n{chunk.text}"
        for chunk in retrieved_chunks
    )

    try:
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
        )
    except Exception as exc:
        raise RAGError(
            "Ollama could not be reached. Make sure Ollama is running."
        ) from exc

    answer = response["message"]["content"].strip()
    return answer or NO_INFORMATION_MESSAGE
