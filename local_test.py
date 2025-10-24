"""
raget_local_test.py

Minimal example to test Giskard RAG Evaluation Toolkit (RAGET)
with a local Llama 3 1-8B-Instruct model in VS Code.

Requirements:
    pip install "giskard[llm]" transformers sentence-transformers
"""

# --------------------------------------------------------------
# Step 0: Imports
# --------------------------------------------------------------
import pandas as pd
from giskard.rag import KnowledgeBase, AgentAnswer, generate_testset, evaluate, QATestset

# Optional: For Hugging Face local model
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --------------------------------------------------------------
# Step 1: Create Knowledge Base
# --------------------------------------------------------------
def create_knowledge_base():
    """Create a small toy knowledge base."""
    docs = [
        "Python is a programming language used for data science.",
        "Giskard helps test and evaluate AI and RAG systems.",
        "Llama 3 is a large language model developed by Meta."
    ]
    df_kb = pd.DataFrame({"text": docs})
    kb = KnowledgeBase.from_pandas(df_kb, columns=["text"])
    return kb, docs

# --------------------------------------------------------------
# Step 2: Load local Llama 3 1-8B-Instruct model
# --------------------------------------------------------------
def load_llama_local(model_path: str):
    """Load local Llama 3 model using Hugging Face Transformers."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return text_gen

# --------------------------------------------------------------
# Step 3: Define RAG prediction function
# --------------------------------------------------------------
def build_rag_predict_fn(text_gen, docs):
    """Return a function suitable for Giskard RAG evaluation."""
    def rag_predict_fn(question: str, history=None) -> AgentAnswer:
        context = "\n".join(docs)
        prompt = f"Answer the question using the following context:\n\n{context}\n\nQ: {question}\nA:"
        output = text_gen(prompt, max_new_tokens=200)
        answer = output[0]["generated_text"]
        return AgentAnswer(message=answer, documents=docs)
    return rag_predict_fn

# --------------------------------------------------------------
# Step 4: Generate synthetic test set
# --------------------------------------------------------------
def generate_rag_testset(kb):
    """Create a small test set automatically."""
    testset = generate_testset(
        knowledge_base=kb,
        num_questions=5,
        language="en",
        agent_description="A local assistant using Llama 3 to answer questions."
    )
    testset.save("rag_testset_local.jsonl")
    return testset

# --------------------------------------------------------------
# Step 5: Evaluate RAG pipeline
# --------------------------------------------------------------
def evaluate_rag_pipeline(rag_predict_fn, testset, kb):
    """Run Giskard evaluation and save HTML report."""
    report = evaluate(
        predict_fn=rag_predict_fn,
        testset=testset,
        knowledge_base=kb
    )
    report.to_html("rag_report.html")
    print("✅ Evaluation complete! Report saved → rag_report.html")

# --------------------------------------------------------------
# Main function to run everything
# --------------------------------------------------------------
def main():
    # Step 1: Knowledge Base
    kb, docs = create_knowledge_base()
    print("✅ Knowledge base created with", len(docs), "documents.")

    # Step 2: Load local model
    model_path = "/path/to/llama-3.1-8b-instruct"  # <-- CHANGE THIS
    text_gen = load_llama_local(model_path)
    print("✅ Local Llama 3 model loaded.")

    # Step 3: Build RAG predict function
    rag_predict_fn = build_rag_predict_fn(text_gen, docs)
    print("✅ RAG prediction function ready.")

    # Step 4: Generate test set
    testset = generate_rag_testset(kb)
    print("✅ Test set generated with", len(testset), "questions.")

    # Step 5: Evaluate
    evaluate_rag_pipeline(rag_predict_fn, testset, kb)

# --------------------------------------------------------------
# Entry point
# --------------------------------------------------------------
if __name__ == "__main__":
    main()
