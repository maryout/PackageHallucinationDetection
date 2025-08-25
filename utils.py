import os
import re
import json
from config import get_data_paths

#Load prompts for a given language from multiple data sources
def load_prompts(language):
    data_paths = get_data_paths(language)
    prompts = []

    for file_path in data_paths['prompts']:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    prompt_text = extract_prompt_from_item(item)
                    if prompt_text:
                        prompts.append(prompt_text)
                except json.JSONDecodeError:
                    prompts.append(line)

    print(f"Loaded {len(prompts)} prompts for {language}")
    return prompts

# Extract prompt text from different possible JSON structures
def extract_prompt_from_item(item):
    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        if "generated_prompt" in item:
            return item["generated_prompt"]
        if "prompt" in item:
            return item["prompt"]
        if "Title" in item and "Body" in item:
            clean_body = re.sub(r'<[^>]+>', '', item["Body"])
            return f"{item['Title'].strip()} - {clean_body.strip()}"
        if "text" in item:
            return item["text"]

    return None

import json
import os

def parse_results_file(file_path):
    """
    Parse a JSONL results file and return a list of result dictionaries.
    """
    results = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results

#For Compute hallucination stats per heuristic from results
def compute_heuristic_stats(results, valid_packages_set):
   
    stats = {f"heuristic{i}": {"valid": set(), "hallucinated": set()} for i in range(1, 4)}

    for r in results:
        pb = r.get("analysis", {}).get("package_breakdown", {})
        for i in range(1, 4):
            heuristic_key = f"heuristic{i}_packages"
            pkgs = pb.get(heuristic_key, [])
            for pkg in pkgs:
                if pkg.lower() in valid_packages_set:
                    stats[f"heuristic{i}"]["valid"].add(pkg)
                else:
                    stats[f"heuristic{i}"]["hallucinated"].add(pkg)

    # Compute rates
    for h in stats:
        total = len(stats[h]["valid"]) + len(stats[h]["hallucinated"])
        stats[h]["rate"] = (len(stats[h]["hallucinated"]) / total) if total > 0 else 0.0

    return stats


def save_heuristic_stats(stats, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    summary = {}

    for h, data in stats.items():
        valid_path = os.path.join(output_dir, f"{h}_valid.txt")
        halluc_path = os.path.join(output_dir, f"{h}_hallucinated.txt")

        with open(valid_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(data["valid"])))

        with open(halluc_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(data["hallucinated"])))

        summary[h] = {
            "valid_count": len(data["valid"]),
            "hallucinated_count": len(data["hallucinated"]),
            "hallucination_rate": data["rate"]
        }

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary

