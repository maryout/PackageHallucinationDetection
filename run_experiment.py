import os
import sys
import time
import json
import argparse
from dotenv import load_dotenv
from detector import PackageHallucinationDetector
from utils import load_prompts
from config import MODELS_CONFIG, SUPPORTED_LANGUAGES, EXPERIMENT_CONFIG


class ExperimentRunner:
    def __init__(self, api_key_dict, language, max_prompts=None):
        self.api_key_dict = api_key_dict
        self.language = language
        self.max_prompts = max_prompts or EXPERIMENT_CONFIG['max_prompts']
        self.detector = None
        self.results = []

    def initialize_detector(self, model_name):
        cfg = self._get_model_cfg(model_name)
        provider = cfg['provider']
        self.detector = PackageHallucinationDetector(
            self.language, model_name,
            openai_api_key=self.api_key_dict.get('openai'),
            claude_api_key=self.api_key_dict.get('claude'),
            deepseek_api_key=self.api_key_dict.get('deepseek'),
            hf_token=self.api_key_dict.get(provider) if provider not in ['openai', 'claude', 'deepseek'] else None
        )
        if not self.detector.valid_packages:
            raise RuntimeError(f"No valid packages loaded for {self.language}")

    def _get_model_cfg(self, model_name):
        return {**MODELS_CONFIG['commercial'], **MODELS_CONFIG['open_source']}.get(model_name, {})

    def load_prompts(self):
        prompts = load_prompts(self.language)
        if not prompts:
            raise RuntimeError(f"No prompts loaded for {self.language}")
        return prompts

    def run_experiment(self, model_name, prompts):
        results = []
        actual_max = min(len(prompts), self.max_prompts)
        model_cfg = self._get_model_cfg(model_name)
        is_commercial = model_name in MODELS_CONFIG['commercial']

        output_dir = f"results/{self.language}/{model_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "results.jsonl")

        print(f"Testing {model_name} ({self.language}) - {actual_max} prompts")

        for i, prompt in enumerate(prompts[:actual_max]):
            print(f"Prompt {i+1}/{actual_max}", end=" ")
            try:
                generated_code = self.detector.generate_code_from_prompt(prompt)
                analysis = self.detector.analyze_code_sample(generated_code, prompt, model_name, is_commercial)

                result = {
                    'language': self.language,
                    'model': model_name,
                    'prompt': prompt,
                    'prompt_index': i,
                    'generated_code': generated_code,
                    'analysis': analysis
                }

                results.append(result)
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")

                print(f"({len(analysis.get('packages_detected', []))} packages found)")
                if is_commercial:
                    time.sleep(EXPERIMENT_CONFIG['rate_limit_delay'])

            except KeyboardInterrupt:
                print("Interrupted by user")
                return results
            except Exception as e:
                print(f"Error: {e}")
                continue

        print(f"Completed {model_name} ({self.language})")
        return results

    def finalize_experiment(self, model_name):
        if self.results:
            output_dir = f"results/{self.language}/{model_name}"
            if os.path.exists(output_dir):
                self.detector.save_results_and_calculate_rates(output_dir)


# Ajouter pour choisir à enlever avant le commit et ajouter dans la fct main
def display_menu(options, title):
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    while True:
        try:
            choice = int(input(f"Enter your choice (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except ValueError:
            pass
        print("Invalid choice")


def get_api_keys(model_name):
    all_models = {**MODELS_CONFIG['commercial'], **MODELS_CONFIG['open_source']}
    if model_name not in all_models:
        sys.exit(f"Model {model_name} not in MODELS_CONFIG")
    provider = all_models[model_name]['provider']
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    if not api_key:
        sys.exit(f"{provider.upper()}_API_KEY not set in .env")
    return {provider: api_key}


def main():
    parser = argparse.ArgumentParser(description="Run experiment with a specified model")
    all_models = list(MODELS_CONFIG['commercial'].keys()) + list(MODELS_CONFIG['open_source'].keys())
    for m in all_models:
        parser.add_argument(f"--{m}", f"--{m.replace('-', '_')}", action="store_true", dest=m.replace('-', '_'))
    args = parser.parse_args()

    selected = next((m for m in all_models if getattr(args, m.replace('-', '_'), False)), None)
    if not selected:
        print("No model specified. Use one of: " + " ".join([f"--{m}" for m in all_models]))
        sys.exit(1)

    load_dotenv()
    api_keys = get_api_keys(selected)

    try:
        lang = display_menu(SUPPORTED_LANGUAGES, "Select a programming language:")
        runner = ExperimentRunner(api_key_dict=api_keys, language=lang)
        runner.initialize_detector(selected)
        prompts = runner.load_prompts()
        runner.results = runner.run_experiment(selected, prompts)
        runner.finalize_experiment(selected)
    except KeyboardInterrupt:
        print("Hard Stoptted")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
