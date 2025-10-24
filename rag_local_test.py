# rag_local_test_llama_autodiscover.py

"""
Offline RAG Evaluation using Llama3 1-8B-Instruct via locally running Ollama.
Automatically discovers the local listener to avoid hardcoding the port.
"""

import requests
import pandas as pd
import json
from pathlib import Path
from html import escape
import socket

# -------------------------------
# Step 0: Auto-discover Local Listener
# -------------------------------
MODEL_NAME = "llama3.1:8b-instruct"

def discover_listener(default_port=11434, host="127.0.0.1"):
    """
    Attempt to connect to the default Ollama port.
    Returns the listener URL if reachable, else None.
    """
    try:
        with socket.create_connection((host, default_port), timeout=1):
            print(f"✅ Found local Ollama listener at {host}:{default_port}")
            return f"http://{host}:{default_port}"
    except (ConnectionRefusedError, socket.timeout):
        print(f"⚠️ Cannot reach Ollama listener at {host}:{default_port}")
        return None

LOCAL_LISTENER_URL = discover_listener()
if not LOCAL_LISTENER_URL:
    print("❌ Local Ollama listener not found. Start Ollama with the model loaded.")
    exit(1)

# -------------------------------
# Step 1: Local Knowledge Base
# -------------------------------
docs = [
    "Python is a programming language widely used for data science and AI.",
    "Giskard is a framework for testing and evaluating AI and RAG systems.",
    "Llama 3 1-8B-Instruct is a large language model by Meta."
]
df_kb = pd.DataFrame({"text": docs})
print(f"✅ Knowledge base created with {len(docs)} documents.")

# -------------------------------
# Step 2: RAG Prediction Function
# -------------------------------
def ollama_predict_listener(prompt):
    """
    Sends the prompt to the local Ollama listener and returns the model's response.
    """
    payload = {"model": MODEL_NAME, "prompt": prompt, "max_tokens": 512}
    try:
        response = requests.post(f"{LOCAL_LISTENER_URL}/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("completion", data.get("text", "No output from model"))
    except requests.RequestException as e:
        print("❌ Error contacting local listener:", e)
        return "Error: Could not generate response"

def build_rag_predict_fn():
    """
    Returns a RAG prediction function using local listener.
    """
    def rag_predict_fn(question: str):
        context = "\n".join(docs)
        prompt = f"Answer the question using the following context:\n\n{context}\n\nQ: {question}\nA:"
        return ollama_predict_listener(prompt)
    return rag_predict_fn

# -------------------------------
# Step 3: Offline Test Cases
# -------------------------------
testset = [
    {"question": "What is Python?", "expected_answer_contains": "programming language"},
    {"question": "What is Giskard?", "expected_answer_contains": "framework"},
    {"question": "Who created Llama3 1-8B-Instruct?", "expected_answer_contains": "Meta"},
    {"question": "What is Python used for?", "expected_answer_contains": "data science"}
]

# -------------------------------
# Step 4: Offline Evaluation
# -------------------------------
def local_rag_evaluate(predict_fn, testset):
    results = []
    for test in testset:
        q = test["question"]
        expected = test["expected_answer_contains"]
        answer = predict_fn(q)
        passed = expected.lower() in answer.lower()
        results.append({"question": q, "answer": answer, "expected": expected, "passed": passed})
    return results

# -------------------------------
# Step 5: Run Evaluation
# -------------------------------
output_dir = Path("rag_reports")
output_dir.mkdir(exist_ok=True)

print(f"\n🔹 Evaluating model: {MODEL_NAME} via listener")
rag_predict_fn = build_rag_predict_fn()
results = local_rag_evaluate(rag_predict_fn, testset)

# Save results as JSON
json_file = output_dir / "rag_results_llama3.1_8b-instruct_listener.json"
with open(json_file, "w") as f:
    json.dump(results, f, indent=2)

# Save simple HTML report
html_file = output_dir / "rag_report_llama3.1_8b-instruct_listener.html"
with open(html_file, "w") as f:
    f.write(f"<h1>RAG Evaluation Report - {MODEL_NAME} (listener)</h1>\n<table border='1'>")
    f.write("<tr><th>Question</th><th>Answer</th><th>Expected</th><th>Passed</th></tr>")
    for r in results:
        f.write(f"<tr><td>{escape(r['question'])}</td><td>{escape(r['answer'])}</td>"
                f"<td>{escape(r['expected'])}</td><td>{r['passed']}</td></tr>")
    f.write("</table>")

print(f"✅ Evaluation complete. JSON: {json_file} | HTML: {html_file}")

# -------------------------------
# Step 6: Quick Validation
# -------------------------------
sample_question = "What is Giskard?"
answer = rag_predict_fn(sample_question)
print(f"\nSample answer from {MODEL_NAME} (listener):\n{answer}")
