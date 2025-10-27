from giskard import Model
import ollama

# Define a simple prediction function using Ollama
def local_llama_predict(texts: list[str]) -> list[str]:
    responses = []
    for text in texts:
        response = ollama.chat(model="llama3", messages=[{"role": "user", "content": text}])
        responses.append(response["message"]["content"])
    return responses

# Wrap it for Giskard
llama_model = Model(
    model=local_llama_predict,
    model_type="text_generation",
    name="Local Llama 3 (Ollama)"
)

print("✅ Local Llama model wrapped successfully!")
