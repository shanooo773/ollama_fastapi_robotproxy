import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.rag_pipeline import rag_answer

# Mix of KB-answerable questions, one deliberately unanswerable question (tests refusal
# instead of hallucination), and small-talk questions (tests the router sending queries
# to the direct-chat path instead of RAG when nothing in the KB is relevant).
TEST_QUESTIONS = [
    "What does NovaTech Robotics do?",
    "How much does the NovaBot Assembly Arm cost?",
    "What is your return policy if I need to cancel an order?",
    "Does NovaFleet AMR store camera footage?",
    "What is the CEO's favorite food?",  # not in the KB — should be refused, not hallucinated
    "Can you customize the assembly arm for my product line?",
    "Hi there! How's your day going?",  # small talk — router should skip RAG
    "Nice robot design, by the way!",  # small talk — router should skip RAG
]


async def main():
    settings = get_settings()
    lines = [f"# RAG Evaluation Report\n\nGenerated: {datetime.now(timezone.utc).isoformat()}\n"]

    for i, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] {question}")
        result = await rag_answer(settings, question)

        route = "RAG (grounded in KB)" if result["used_rag"] else "Direct chat (router skipped RAG)"

        lines.append(f"## Q{i}: {question}\n")
        lines.append(f"**Router decision:** {route}\n")
        if settings.enable_query_rewrite:
            lines.append(f"**Rewritten search query:** {result['search_query']}\n")
        lines.append(f"**Answer:** {result['answer']}\n")
        lines.append(f"**Timings (s):** {result['timings']}\n")

        if result["used_rag"]:
            lines.append(f"**Sources retrieved:** {', '.join(result['sources']) or 'none'}\n")
            lines.append("**Retrieved chunks (with distance, lower = more relevant):**\n")
            for src, dist, chunk in zip(result["sources"], result["distances"], result["retrieved_chunks"]):
                preview = chunk[:180] + ("..." if len(chunk) > 180 else "")
                lines.append(f"- `{src}` (distance={dist:.4f}): {preview}\n")
        else:
            best = min(result["distances"]) if result["distances"] else None
            lines.append(f"**Best retrieval distance was {best:.4f} — above threshold "
                          f"({settings.rag_distance_threshold}), so RAG was skipped.**\n" if best is not None
                          else "**No chunks in the collection at all.**\n")

        lines.append("\n---\n")

    report_path = Path(__file__).resolve().parent.parent / "rag_evaluation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {report_path}")
    print(f"Structured event log at: {settings.rag_log_path}")


if __name__ == "__main__":
    asyncio.run(main())
