import requests
from giskard.rag import AgentAnswer

# === Your internal RAG front-end URL ===
RAG_URL = "https://your-internal-rag-frontend.com/api/query"  # Replace with your API endpoint

# === Function to query RAG ===
def answer_fn(question: str, history=None) -> AgentAnswer:
    payload = {"query": question}
    response = requests.post(RAG_URL, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    answer_text = data.get("answer") or data.get("response") or ""
    contexts = data.get("contexts", [])

    return AgentAnswer(message=answer_text, documents=contexts)

# === List of sample questions ===
questions = [
    "What is 2 + 2?",
    "Who wrote 1984?",
    "What color is the sky?",
]

# === Run queries and print answers ===
for q in questions:
    ans = answer_fn(q)
    print(f"Q: {q}\nA: {ans.message}\nContexts: {ans.documents}\n---")
