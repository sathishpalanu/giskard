# internal_rag_eval_localapi.py
"""
Offline Internal RAG Evaluation using Giskard SDK
Leverages:
- Your internal Front End / Light RAG endpoints
- Local Llama API at http://localhost:11434 as judge
Fully offline, suitable for restricted corporate environments
"""

import requests
import pandas as pd
from giskard import Dataset, TestSuite

# ------------------------------
# 1. Configuration
# ------------------------------
FRONTEND_URL = "http://localhost:3000/api/query"   # Your front end API
LIGHT_RAG_URL = "http://localhost:8000/retrieve"  # Your Light RAG API

# Local Llama judge API
LOCAL_JUDGE_API = "http://localhost:11434"

# Example test queries for your internal documentation
test_queries = [
    {
        "query": "Where is the Architecture Decision Record Template?",
        "expected_document": "Architecture Decision Record Template"
    },
    {
        "query": "Where is the Reference Architecture Template?",
        "expected_document": "Reference Architecture Template"
    },
    {
        "query": "Show me the recent Analysis of Alternatives?",
        "expected_document": "Recent Analysis of Alternatives Document"
    }
]

# Choose which endpoint to use
USE_LIGHT_RAG = True

# ------------------------------
# 2. Judge Function using Local API
# ------------------------------
def llama_judge(query, retrieved_doc):
    """
    Judge retrieved document correctness using local Llama API
    Returns 1 if likely correct, 0 otherwise
    """
    payload = {
        "query": query,
        "retrieved_document": retrieved_doc
    }
    try:
        resp = requests.post(LOCAL_JUDGE_API, json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json().get("judge", "No")
        return 1 if str(result).lower() == "yes" else 0
    except Exception as e:
        print(f"Error calling local judge API: {e}")
        return 0

# ------------------------------
# 3. Query Internal RAG Endpoint
# ------------------------------
def retrieve_from_rag(query):
    url = LIGHT_RAG_URL if USE_LIGHT_RAG else FRONTEND_URL
    try:
        resp = requests.post(url, json={"query": query}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("documents", ["No document"])
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return ["No document"]

# ------------------------------
# 4. Build Giskard Dataset
# ------------------------------
dataset_records = []

for q in test_queries:
    retrieved_docs = retrieve_from_rag(q["query"])
    first_doc = retrieved_docs[0] if retrieved_docs else "No document"
    score = llama_judge(q["query"], first_doc)
    dataset_records.append({
        "query": q["query"],
        "expected_document": q["expected_document"],
        "retrieved_doc": first_doc,
        "judge_score": score
    })

dataset = Dataset.from_dicts(
    dataset_records,
    name="Internal RAG Evaluation",
    target="expected_document",
    prediction="retrieved_doc"
)

print("Dataset Preview:")
print(dataset.df)

# ------------------------------
# 5. Run Giskard Test Suite
# ------------------------------
test_suite = TestSuite(name="Internal RAG Test Suite")

for _, row in dataset.df.iterrows():
    test_suite.add_test(
        f"Judge Query: {row['query']}",
        lambda r=row: r["judge_score"]
    )

results = test_suite.run()
print("\nGiskard Test Suite Results:")
print(results)

# ------------------------------
# 6. Save CSV Report
# ------------------------------
df = pd.DataFrame(dataset.df)
df.to_csv("internal_rag_eval_results.csv", index=False)
print("\nOffline CSV report saved: internal_rag_eval_results.csv")
