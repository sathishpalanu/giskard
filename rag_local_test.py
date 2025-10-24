# rag_local_test.py

"""
Fully local RAG evaluation using Giskard Python SDK and Ollama models.
Works on Mac M1 with Llama3 1-8B-Instruct and Phi3.5:latest.
"""

import pandas as pd
from giskard.rag import KnowledgeBase, AgentAnswer, generate_testset, evaluate
import subprocess

# -------------------------------
# Step 1: Create Local Knowledge Base
# -------------------------------

docs = [
    "Python is a programming language widely used for data science and AI.",
    "Giskard is a framework for testing and evaluating AI and RAG systems.",
    "Llama 3 1-8B-Instruct is a large language model by Meta.",
    "Phi 3.5 is a local LLM model for instruction-following tasks."
]

df_kb = pd.DataFrame({"text": docs})
kb = KnowledgeBase.from_pandas(df_kb, columns=["text"])
print(f"✅ Knowledge base created with {len(docs)} documents.")

# -------------------------------
# Step 2: RAG Prediction Function
# -------------------------------

def ollama_predict(prompt, model_name="llama3.1:8b-instruct"):
    """
    Call Ollama CLI to generate a response from a local model.
    """
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
    """
    Returns a RAG-compatible prediction function for Giskard.
    """
    def rag_predict_fn(question: str, history=None):
        context = "\n".join(docs)
        prompt = f"Answer the question using the following context:\n\n{context}\n\nQ: {question}\nA:"
        answer = ollama_predict(prompt, model_name=model_name)
        return AgentAnswer(message=answer, documents=docs)
    return rag_predict_fn

# -------------------------------
# Step 3: Generate Synthetic Test Cases
# -------------------------------

testset = generate_testset(
    knowledge_base=kb,
    num_questions=5,
    language="en",
    agent_description="Local assistant using Llama3 or Phi3.5 via Ollama"
)
testset.save("rag_testset_local.jsonl")
print(f"✅ Test set generated with {len(testset)} questions.")

# -------------------------------
# Step 4: Evaluate RAG Pipeline
# -------------------------------

models = ["llama3.1:8b-instruct", "phi3.5:latest"]

for model_name in models:
    print(f"\n🔹 Evaluating model: {model_name}")
    rag_predict_fn = build_rag_predict_fn(model_name=model_name)
    
    report = evaluate(
        predict_fn=rag_predict_fn,
        testset=testset,
        knowledge_base=kb
    )
    
    html_file = f"rag_report_{model_name.replace(':','_')}.html"
    report.to_html(html_file)
    print(f"✅ RAG evaluation complete for {model_name}. Report saved → {html_file}")

# -------------------------------
# Step 5: Quick Validation
# -------------------------------

sample_question = "What is Giskard?"
for model_name in models:
    rag_predict_fn = build_rag_predict_fn(model_name=model_name)
    answer = rag_predict_fn(sample_question)
    print(f"\nSample answer from {model_name}:\n{answer.message}")
