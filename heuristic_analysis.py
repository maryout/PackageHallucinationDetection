#!/usr/bin/env python3
import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from utils import parse_results_file, compute_heuristic_stats, save_heuristic_stats

def load_valid_packages(file_path):
    """Load valid package names from a file into a set."""
    valid_packages = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                package = line.strip()
                if package:
                    valid_packages.add(package)
        return valid_packages
    except FileNotFoundError:
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
    except Exception as e:
        raise Exception(f"Erreur lors de la lecture de {file_path}: {str(e)}")

def plot_heuristic_comparison(summary, model_name, output_dir):
    heuristics = list(summary.keys())
    rates = [summary[h]["hallucination_rate"] * 100 for h in heuristics]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(heuristics, rates, color=["#4CAF50", "#FFC107", "#F44336"])
    plt.title(f"Hallucination Rates by Heuristic - {model_name}")
    plt.ylabel("Hallucination Rate (%)")
    plt.ylim(0, 100)
    for bar, rate in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{rate:.2f}%", ha="center", fontsize=10)
    plot_path = os.path.join(output_dir, "heuristics_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def main():
    parser = argparse.ArgumentParser(description="Analyze heuristic hallucination rates")
    parser.add_argument("--model", required=True, help="Model name (e.g., claude-3-haiku-20240307)")
    args = parser.parse_args()

    # Model name validation
    invalid_chars = r'[<>:"/\\|?*]'
    if re.search(invalid_chars, args.model):
        raise ValueError(f"Le nom du modèle '{args.model}' contient des caractères invalides ({invalid_chars}). Veuillez utiliser un nom valide.")

    # Input and output directories
    input_dir = f"results/javascript/{args.model}"
    output_base = f"results_analysis/"
    model_dir = os.path.join(output_base, args.model)
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Le dossier {input_dir} n'existe pas. Veuillez créer ce dossier et y placer les fichiers .jsonl.")

    os.makedirs(model_dir, exist_ok=True)

    # Find the results.jsonl file
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]
    if not jsonl_files:
        raise FileNotFoundError(f"Aucun fichier .jsonl trouvé dans {input_dir}")


    results_file = os.path.join(input_dir, "results.jsonl")
    if not os.path.exists(results_file):
        results_file = os.path.join(input_dir, jsonl_files[0])

    print(f"Parsing results file: {results_file}...")
    results = parse_results_file(results_file)

    
    valid_packages_file = "data/javascript/master/valid-npm.csv" 
    print(f"Loading valid packages from {valid_packages_file}...")
    valid_packages_set = load_valid_packages(valid_packages_file)

    stats = compute_heuristic_stats(results, valid_packages_set)
    print(f"Saving stats to {model_dir}...")
    summary = save_heuristic_stats(stats, model_dir)

    print("Generating plot....")
    plot_path = plot_heuristic_comparison(summary, args.model, model_dir)

    print(f"Analysis completed. Summary and plot saved in {model_dir}")
    print(f"Plot path: {plot_path}")

if __name__ == "__main__":
    main()