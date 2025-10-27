import os
import requests
import pandas as pd

from giskard.rag import (
    KnowledgeBase,
    generate_testset,
    QATestset,
    evaluate,
    AgentAnswer,
)
from giskard.rag.metrics.ragas_metrics import (
    # pick the metrics you want
    ragas_context_precision,
    ragas_context_recall
)

# --- CONFIGURE ---
# Your frontend URL
RAG_URL = "https://YOUR-RAG-APP.com/api/query"

# (Optional) your knowledge base documents (if you want auto testset generation)
KB_CSV_PATH = "path/to/your_kb_docs.csv"
# Columns in your CSV that contain the text content
KB_COLUMNS = ["text"]  # adapt as needed

# Agent description (for test-set generation)
AGENT_DESCRIPTION = "A chatbot answering questions about my domain"

# Number of questions to generate
NUM_QUESTIONS = 50

# --- setup LLM client if needed ---
# e.g., using OpenAI:
os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"
# optionally set embedding model / llm via giskard.llm.set_* if needed

# --- Build knowledge base ---
df_kb = pd.read_csv(KB_CSV_PATH)
kb = KnowledgeBase.from_pandas(df_kb, columns=KB_COLUMNS)

# --- Generate testset (if you don’t already have one) ---
testset = generate_testset(
    knowledge_base=kb,
    num_questions=NUM_QUESTIONS,
    language='en',
    agent_description=AGENT_DESCRIPTION
)
# Save it for reuse
testset.save("rag_testset.jsonl")

# --- Or if you already have a testset ---
# testset = QATestset.load("rag_testset.jsonl")

# --- Define prediction function that uses your frontend URL ---
def answer_fn(question: str, history: list[dict] = None) -> AgentAnswer:
    """
    Calls the RAG front-end endpoint and returns an AgentAnswer
    which includes the answer and optionally the context docs retrieved by the agent.
    """
    payload = {"query": question}
    # if your API expects conversation history or other fields, modify this
    resp = requests.post(RAG_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    # Extract answer
    answer_text = data.get("answer", "")
    
    # If your API returns retrieved context (chunks) do:
    contexts = data.get("contexts", [])  # adjust key if different
    # Ensure contexts is a list of strings
    docs = [str(c) for c in contexts]
    
    return AgentAnswer(message=answer_text, documents=docs)

# --- Run evaluation ---
report = evaluate(
    answer_fn,
    testset=testset,
    knowledge_base=kb,
    metrics=[ragas_context_precision, ragas_context_recall]
)

# --- Inspect & save report ---
report.save("rag_report")
html = report.to_html(embed=True)
with open("rag_report.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Evaluation completed. Report saved to rag_report.html")
