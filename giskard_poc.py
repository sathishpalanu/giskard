import pandas as pd
import giskard
from giskard import Model, Dataset, test

# --- Dummy model (simulates an LLM) ---
def dummy_predict(inputs):
    return ["This is a simple response to: " + q for q in inputs]

# --- Sample dataset ---
df = pd.DataFrame({
    "query": ["Hello", "Tell me a joke", "What is AI?"],
    "expected_answer": ["Hi there!", "Funny!", "Artificial Intelligence"]
})

dataset = Dataset.from_pandas(df, target="expected_answer", column_types={"query": "text"})

# --- Register model ---
model = Model(prediction_function=dummy_predict, model_type="text_generation", name="DummyLLM")

# --- Run a single built-in test ---
suite = giskard.TestSuite(name="POC suite")
suite.add_test(test.test_robustness_perturbation(model, dataset))
result = suite.run()

# --- Save & show results ---
result.to_html("giskard_report.html")
print("\n✅ POC finished! Open 'giskard_report.html' to view your results.\n")
