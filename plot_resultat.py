import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Dossier des résultats
RESULT_DIR = "results"
LANGUAGES = ["python", "javascript"]

def extract_hallucination_rate(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    match = re.search(r"Hallucination rate:\s*([\d.]+)%", text)
    return float(match.group(1)) if match else None

def load_results():
    results = {lang: {} for lang in LANGUAGES}
    for lang in LANGUAGES:
        lang_dir = os.path.join(RESULT_DIR, lang)
        if not os.path.isdir(lang_dir):
            continue
        for model in os.listdir(lang_dir):
            model_dir = os.path.join(lang_dir, model)
            report_file = os.path.join(model_dir, f"{lang}_final_report.txt")
            if os.path.isfile(report_file):
                rate = extract_hallucination_rate(report_file)
                if rate is not None:
                    results[lang][model] = rate
    return results

def plot_results(results):
    models = sorted(set(results["python"].keys()) | set(results["javascript"].keys()))
    python_rates = [results["python"].get(m, 0) for m in models]
    js_rates = [results["javascript"].get(m, 0) for m in models]
    
    y_pos = np.arange(len(models))
    
    plt.figure(figsize=(10,6))
    plt.barh(y_pos - 0.2, python_rates, height=0.4, label="Python")
    plt.barh(y_pos + 0.2, js_rates, height=0.4, label="JavaScript")
    
    plt.yticks(y_pos, models)
    plt.xlabel("Hallucination Rate (%)")
    plt.title("Hallucination Rate per Model and Language")
    plt.legend()
    plt.tight_layout()
    
    # jo 

    plt.savefig("hallucination_rates.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    results = load_results()
    if any(results.values()):
        plot_results(results)
    else:
        print("Empty 'results'.")
