import requests
from giskard.rag import QATestset, AgentAnswer, evaluate
from giskard.rag.metrics.ragas_metrics import (
    ragas_context_precision,
    ragas_context_recall,
)

# === 1️⃣ YOUR FRONTEND URL ===
RAG_URL = "https://your-rag-frontend-url.com/api/query"  # 👈 Replace with your actual chat API URL


# === 2️⃣ DEFINE A FEW TEST QUESTIONS (Ground Truths) ===
# You can start small — 2 or 3 questions with known correct answers.
questions = [
    "What is the capital of France?",
    "Who wrote the novel 1984?",
]
ground_truths = [
    "Paris",
    "George Orwell",
]

# Build a very basic testset
testset = QATestset.from_dict({
    "question": questions,
    "ground_truth": ground_truths
})


# === 3️⃣ DEFINE HOW TO CALL YOUR RAG FRONTEND ===
def answer_fn(question: str, history=None) -> AgentAnswer:
    """
    Sends a user query to your RAG front-end and returns the answer.
    If your API returns retrieved docs, include them; otherwise, pass [].
    """
    payload = {"query": question}
    response = requests.post(RAG_URL, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    # Adjust these keys to match your API response
    answer_text = data.get("answer") or data.get("response") or ""
    contexts = data.get("contexts", [])  # optional

    return AgentAnswer(message=answer_text, documents=contexts)


# === 4️⃣ RUN A SIMPLE EVALUATION ===
# Here we’ll just check context precision/recall — no need for a KB yet.
report = evaluate(
    answer_fn,
    testset=testset,
    metrics=[ragas_context_precision, ragas_context_recall],
)

# === 5️⃣ SAVE OR PRINT RESULTS ===
report.save("basic_rag_report")
print(report.to_pandas())
print("\n✅ RAG Evaluation complete. See saved report in 'basic_rag_report' folder.")
