"""
raget_local_test_ollama.py

Minimal example to test Giskard RAG Evaluation Toolkit (RAGET)
with a local Ollama Llama 3 1-8B-Instruct model.

Requirements:
    pip install giskard ollama
"""

import pandas as pd
from giskard.rag import KnowledgeBase, AgentAnswer, generate_testset, evaluate

# --------------------------------------------------------------
# Step 1: Knowledge Base
# --------------------------------------------------------------
def create_knowledge_base():
    docs = [
        "Python is a programming language used for data science.",
        "Giskard helps test and evaluate AI and RAG systems.",
        "Llama 3 is a large language model developed by Meta."
    ]
    df_kb = pd.DataFrame({"text": docs})
    kb = KnowledgeBase.from_pandas(df_kb, columns=["text"])
    return kb, docs

# --------------------------------------------------------------
# Step 2: Load Ollama model
# --------------------------------------------------------------
def load_ollama_model(model_name="llama3.1:8b-instruct"):
    """
    Returns a function that calls Ollama CLI to generate text.
    """
    import subprocess

    def predict(prompt):
        # Use Ollama CLI to generate output
        result = subprocess.run(
            ["ollama", "generate", model_name, prompt],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()

    return predict

# --------------------------------------------------------------
# Step 3: RAG prediction function
# --------------------------------------------------------------
def build_rag_predict_fn(ollama_predict, docs):
    def rag_predict_fn(question: str, history=None):
        context = "\n".join(docs)
        prompt = f"Answer the question using the following context:\n\n{context}\n\nQ: {question}\nA:"
        answer = ollama_predict(prompt)
        return AgentAnswer(message=answer, documents=docs)
    return rag_predict_fn

# --------------------------------------------------------------
# Step 4: Generate test set
# --------------------------------------------------------------
def generate_rag_testset(kb):
    testset = generate_testset(
        knowledge_base=kb,
        num_questions=5,
        language="en",
        agent_description="A local assistant using Llama 3 via Ollama."
    )
    testset.save("rag_testset_local.jsonl")
    return testset

# --------------------------------------------------------------
# Step 5: Evaluate RAG pipeline
# --------------------------------------------------------------
def evaluate_rag_pipeline(rag_predict_fn, testset, kb):
    report = evaluate(
        predict_fn=rag_predict_fn,
        testset=testset,
        knowledge_base=kb
    )
    report.to_html("rag_report.html")
    print("✅ Evaluation complete! Report saved → rag_report.html")

# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main():
    kb, docs = create_knowledge_base()
    print("✅ Knowledge base created with", len(docs), "documents.")

    ollama_predict = load_ollama_model()
    print("✅ Ollama Llama 3 model ready.")

    rag_predict_fn = build_rag_predict_fn(ollama_predict, docs)
    print("✅ RAG prediction function ready.")

    testset = generate_rag_testset(kb)
    print("✅ Test set generated with", len(testset), "questions.")

    evaluate_rag_pipeline(rag_predict_fn, testset, kb)

# --------------------------------------------------------------
if __name__ == "__main__":
    main()
