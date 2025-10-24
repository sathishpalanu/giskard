# rag_local_test_llama3_cli.py

"""
Offline RAG Evaluation using Llama3 1-8B-Instruct via Ollama CLI.
This avoids REST calls and works fully locally on Mac M1.
"""

import subprocess
import pandas as pd
import json
from pathlib import Path
from html import escape

# -------------------------------
# Step 0: Check Ollama CLI
# -------------------------------
def check_ollama_cli():
    """Verify Ollama CLI is installed and reachable."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        print("✅ Ollama CLI found. Installed models:\n", result.stdout)
    except FileNotFoundError:
        print("❌ Ollama CLI not found. Install Ollama App and ensure `ollama` is in PATH.")
        exit(1)
    except subprocess.CalledProcessError as e:
        print("❌ Error running Ollama CLI:", e)
        exit(1)

check_ollama_cli()

MODEL_NAME = "llama3.1:8b-instruct"

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
# Step 2: RAG Prediction Function via CLI
# -------------------------------
def ollama_cli_predict(prompt: str) -> str:
    """
    Call Ollama CLI to generate a completion using the local model.
    """
    try:
        result = subprocess.run(
            ["ollama", "generate", MODEL_NAME, prompt],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("❌ Error calling Ollama CLI:", e)
        return "Error: Could not generate response"

def build_rag_predict_fn():
    """
    Returns a RAG prediction function using Ollama CLI.
    """
    def rag_predict_fn(question: str):
        context = "\n".join(docs)
        prompt = f"Answer the question using the following context:\n\n{context}\n\nQ: {question}\nA:"
        return ollama_cli_predict(prompt)
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

print(f"\n🔹 Evaluating model: {MODEL_NAME} via Ollama CLI")
rag_predict_fn = build_rag_predict_fn()
results = local_rag_evaluate(rag_predict_fn, testset)

# Save results as JSON
json_file = output_dir / "rag_results_llama3.1_8b-instruct_cli.json"
with open(json_file, "w") as f:
    json.dump(results, f, indent=2)

# Save simple HTML report
html_file = output_dir / "rag_report_llama3.1_8b-instruct_cli.html"
with open(html_file, "w") as f:
    f.write(f"<h1>RAG Evaluation Report - {MODEL_NAME} (CLI)</h1>\n<table border='1'>")
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
print(f"\nSample answer from {MODEL_NAME} (CLI):\n{answer}")
