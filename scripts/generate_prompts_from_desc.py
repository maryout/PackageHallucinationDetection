import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"  # Hugging Face model
INPUT_FILE = "../data/master/top5-pkg_desc_pypi.jsonl"
OUTPUT_FILE = "../data/prompts/generated-py-prompts.jsonl"
N_PACKAGES = 5

print("📥 Loading CodeLlama...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

    for i, line in enumerate(f_in):
        if i >= N_PACKAGES:
            break

        item = json.loads(line)
        pkg = item["package_name"]
        desc = item["description"]

        print(f"➡️ Generating coding prompt for {pkg}...")

        prompt = (
            f"Generate Python code that uses the package '{pkg}'. "
            f"Description: {desc}"
        )

        try:
            result = generator(prompt, max_new_tokens=100, do_sample=True, top_k=50)
            gen_prompt = result[0]["generated_text"]
            print(f"✅ Success for {pkg}: {gen_prompt[:60]}...")
        except Exception as e:
            gen_prompt = f"ERROR: {e}"
            print(f"⚠️ Error for {pkg}: {e}")

        # Always write something
        json.dump({
            "id": i,
            "package": pkg,
            "description": desc,
            "generated_prompt": gen_prompt
        }, f_out, ensure_ascii=False)
        f_out.write("\n")

print(f"✅ Saved up to {N_PACKAGES} prompts to {OUTPUT_FILE}")
