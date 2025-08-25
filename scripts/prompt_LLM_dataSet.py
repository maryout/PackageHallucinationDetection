import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file. Please set it in the .env file.")

#LANGUAGE = "python"
LANGUAGE = "javascript"
INPUT_FILE = f"../data/{LANGUAGE}/master/top250-pkg_desc_pypi.jsonl"
OUTPUT_FILE = f"../data/{LANGUAGE}/prompts/LLM_PKG_DESC.jsonl"
N_PACKAGES = 250

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_prompt(description):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"Create a single sentence prompt that starts with 'Generate {LANGUAGE} code that' based on the package description. Be specific and detailed about what the code should accomplish."
                },
                {
                    "role": "user",
                    "content": f"Package description: {description}"
                }
            ],
            max_tokens=150,
            temperature=0.7,
            timeout=30
        )
        
        generated = response.choices[0].message.content.strip()
        
        if not generated.startswith(f"Generate {LANGUAGE} code that"):
            generated = f"Generate {LANGUAGE} code that {generated}"
        
        return generated
        
    except Exception as e:
        print(f"Error: {e} - Using fallback")
        return f"Generate {LANGUAGE} code that implements functionality described as: {description}"

with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        if i >= N_PACKAGES:
            break
        
        print(f"{i+1}: ", end="")
        
        try:
            item = json.loads(line)
            print(f"{item['package_name']}", end=" -> ")
            
            prompt = generate_prompt(item["description"])
            fout.write(f'"{prompt}"\n')
            fout.flush()
            
            print("Done")
            time.sleep(0.5)
            
        except KeyboardInterrupt:
            print("Stopped")
            break
        except:
            print("Skipped")
            continue

print("Finished!")