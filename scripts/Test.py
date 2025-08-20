from transformers import pipeline


# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="TheBloke/CodeLlama-7B-Instruct-GGUF" , use_auth_token="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" )  # replace with your model name or path


# Example prompt: generate Python code to compute factorial
prompt = "Write a Python function that calculates the factorial of a number."

# Generate output
output = pipe(prompt, max_new_tokens=100)  # limit output tokens for speed

# Print the generated code
print(output[0]['generated_text'])


