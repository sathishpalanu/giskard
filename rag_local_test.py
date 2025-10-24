import subprocess
import pandas as pd
import json
from pathlib import Path
from html import escape

# -------------------------------
# Step 0: Local Knowledge Base
# -------------------------------
docs = [
    "Python is a programming language widely used for data science and AI.",
    "Giskard is a framework for testing and evaluating AI and RAG systems.",
    "Llama 3 1-8B-Instruct is a large language model by Meta.",
    "Phi 3.5 is a local LLM model for instruction-following tasks."
]

df_kb = pd.DataFrame({"text": docs})
print(f"✅ Knowledge base created with {len(docs)} documents.")

# -------------------------------
# Step 1: RAG Prediction Function
# -------------------------------
def ollama_predict(prompt, model_name="llama3.1:8b-instruct"):
    try:
        result = subprocess.run(
            ["ollama", "generate", model_name, prompt],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("❌ Ollama generate failed:", e.stderr)
        return "Error: Could not generate response"

def build_rag_predict_fn(model_name="llama3.1:8b-instruct"):
    def rag_predict_fn(question: str):
        context = "\n".join(docs)
        prompt = f"Answer the question using the following context:\n\n{context}\n\nQ: {question}\nA:"
        answer = ollama_predict(prompt, model_name=model_name)
        return answer
    return rag_predict_fn

# -------------------------------
# Step 2: Offline Test Cases
# -------------------------------
testset = [
    {"question": "What is Python?", "expected_answer_contains": "programming language"},
    {"question": "What is Giskard?", "expected_answer_contains": "framework"},
    {"question": "Who created Llama3 1-8B-Instruct?", "expected_answer_contains": "Meta"},
    {"question": "What is Phi 3.5?", "expected_answer_contains": "local LLM model"},
    {"question": "What is Python used for?", "expected_answer_contains": "data science"}
]

# -------------------------------
# Step 3: Offline Evaluati
