import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# -------- Config --------
# Using a smaller, CPU-friendly model for faster inference
# MODEL_NAME = "microsoft/DialoGPT-small"  # Much faster alternative
MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"  # Hugging Face model
# Alternative options (uncomment to try):
# MODEL_NAME = "microsoft/CodeT5p-220m-py"
# MODEL_NAME = "Salesforce/codet5-small"

INPUT_FILE = "../data/master/top5-pkg_desc_pypi.jsonl"
OUTPUT_FILE = "../data/prompts/generated-py-prompts.jsonl"
N_PACKAGES = 1

# CPU optimization settings
torch.set_num_threads(4)  # Adjust based on your CPU cores
torch.set_grad_enabled(False)  # Disable gradients for inference

# -------- System Message --------
SYSTEM_MESSAGE = (
    "You are a coding assistant that assists users in creating simple prompts that will be used to "
    "generate Python code. No code should be used in the response."
)

def user_message_from_description(desc: str, language: str = "Python") -> str:
    """Create user message from package description"""
    return (
        f"Your answer must begin with 'Generate {language} code that' and must not be longer than one sentence. "
        f"Do not include extra text or formatting. "
        f"Write a prompt that would generate {language} code to accomplish the same tasks as the following "
        f"package description: {desc}"
    )

def make_simple_prompt(system_text: str, user_text: str, desc: str) -> str:
    """Simplified prompt format for smaller models"""
    return f"Task: Create a coding prompt. {desc}. Response:"

def normalize_one_sentence(text: str, language: str = "Python") -> str:
    """Post-process to keep a single sentence starting with the required prefix"""
    if not text:
        return f"Generate {language} code that performs the described functionality."
    
    # Clean the text
    text = text.strip()
    
    # Remove common prefixes that models add
    prefixes_to_remove = [
        "Sure! Here's a prompt:",
        "Here's a prompt:",
        "Response:",
        "Task:",
        "Prompt:",
        "Generate a prompt:",
    ]
    
    for prefix in prefixes_to_remove:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    
    # Take the first line/sentence
    text = text.split("\n")[0].strip()
    
    # Split at sentence enders and take the first sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    first_sentence = sentences[0].strip() if sentences else text.strip()
    
    # Ensure it starts with the correct prefix
    prefix = f"Generate {language} code that"
    if not first_sentence.lower().startswith(prefix.lower()):
        # Clean up the sentence and add prefix
        if first_sentence.startswith('"') and first_sentence.endswith('"'):
            first_sentence = first_sentence[1:-1]
        first_sentence = f"{prefix} {first_sentence}"
    
    # Remove ending punctuation if it's not a period
    if first_sentence.endswith(('.', '!', '?')):
        pass  # Keep sentence endings
    else:
        first_sentence += "."
    
    return first_sentence

def create_fallback_prompt(desc: str, language: str = "Python") -> str:
    """Create a fallback prompt if generation fails"""
    # Simple keyword extraction and prompt creation
    keywords = []
    common_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    
    words = re.findall(r'\b\w+\b', desc.lower())
    for word in words:
        if len(word) > 3 and word not in common_words:
            keywords.append(word)
    
    if keywords:
        key_functionality = ' '.join(keywords[:3])  # Take first 3 keywords
        return f"Generate {language} code that implements {key_functionality} functionality."
    else:
        return f"Generate {language} code that implements the described functionality."

print(f"🔧 Setting up CPU optimizations...")
print(f"📥 Loading model: {MODEL_NAME}")

try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Try different approaches for model loading
    print("🔄 Trying standard model loading...")
    try:
        # First try: Simple loading without device specifications
        generator = pipeline(
            "text-generation",
            model=MODEL_NAME,
            tokenizer=tokenizer,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        print(f"✅ Model loaded successfully with pipeline!")
        
    except Exception as pipeline_error:
        print(f"⚠️ Pipeline loading failed: {pipeline_error}")
        print("🔄 Trying manual model loading...")
        
        # Second try: Load model and tokenizer separately
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        model.eval()
        
        # Create pipeline without device specifications
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
        print(f"✅ Model loaded successfully with manual approach!")
    
except Exception as e:
    print(f"❌ Error loading primary model: {e}")
    print("💡 Switching to GPT-2 fallback...")
    
    try:
        # Fallback to GPT-2 which is more reliable
        MODEL_NAME = "gpt2"
        generator = pipeline(
            "text-generation",
            model=MODEL_NAME,
            torch_dtype=torch.float32
        )
        tokenizer = generator.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ GPT-2 fallback loaded successfully!")
        
    except Exception as fallback_error:
        print(f"❌ Even GPT-2 failed: {fallback_error}")
        print("💡 Trying most basic setup...")
        
        # Last resort: Most basic setup
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print(f"✅ Basic GPT-2 setup successful!")

# Create output directory
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

print(f"🚀 Starting generation for {N_PACKAGES} packages...")

written = 0
with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        if i >= N_PACKAGES:
            break
            
        try:
            item = json.loads(line)
            pkg = item["package_name"]
            desc = item["description"]
            
            print(f"➡️  Processing {i+1}/{N_PACKAGES}: {pkg}")
            
            # Create a simple, effective prompt
            user_text = user_message_from_description(desc, language="Python")
            prompt = f"Create a coding prompt: {desc[:100]}... Answer:"
            
            try:
                # Generate with conservative settings
                outputs = generator(
                    prompt,
                    max_new_tokens=32,      # Reduced for faster generation
                    temperature=0.1,        # Low temperature for consistency
                    do_sample=True,         # Enable sampling for variety
                    top_p=0.9,             # Nucleus sampling
                    repetition_penalty=1.1, # Avoid repetition
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_full_text=False, # Only return generated part
                    clean_up_tokenization_spaces=True
                )
                
                # Extract generated text
                generated_text = outputs[0]["generated_text"].strip()
                
                print(f"   🔍 Raw output: {repr(generated_text[:50])}...")
                
                # Process the output
                processed_prompt = normalize_one_sentence(generated_text, language="Python")
                
            except Exception as gen_error:
                print(f"   ⚠️  Generation failed: {gen_error}")
                processed_prompt = create_fallback_prompt(desc, language="Python")
            
            # Validate the output
            if len(processed_prompt) < 10 or not processed_prompt.startswith("Generate Python code"):
                print(f"   🔧 Using fallback prompt")
                processed_prompt = create_fallback_prompt(desc, language="Python")
            
            # Save the result
            result = {
                "id": i,
                "package": pkg,
                "description": desc,
                "generated_prompt": processed_prompt
            }
            
            json.dump(result, fout, ensure_ascii=False)
            fout.write("\n")
            fout.flush()  # Ensure data is written immediately
            written += 1
            
            print(f"   ✅ Generated: {processed_prompt}")
            
        except Exception as e:
            print(f"   ❌ Error processing line {i}: {e}")
            # Write error entry
            error_result = {
                "id": i,
                "package": "unknown",
                "description": "error",
                "generated_prompt": f"Generate Python code that handles the error: {str(e)}"
            }
            json.dump(error_result, fout, ensure_ascii=False)
            fout.write("\n")
            fout.flush()
            written += 1

print(f"✅ Completed! Wrote {written} lines to {OUTPUT_FILE}")

# Optional: Display sample results
if written > 0:
    print(f"\n📋 Sample results:")
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:  # Show first 3 results
                    break
                item = json.loads(line)
                print(f"   {i+1}. {item['package']}: {item['generated_prompt']}")
    except Exception as e:
        print(f"   Could not display results: {e}")

print(f"\n🎉 Script completed successfully!")