import os
import re
import pandas as pd
from openai import OpenAI
from anthropic import Anthropic
from huggingface_hub import InferenceClient
from config import MODELS_CONFIG, SYSTEM_MESSAGES, get_data_paths


class PackageHallucinationDetector:
    #Detect hallucinated packages using the paper 3 heuristics

    def __init__(self, language, model_key, openai_api_key=None, claude_api_key=None,
                 hf_token=None, deepseek_api_key=None):
        self.language = language
        self.model_key = model_key
        self.data_paths = get_data_paths(language)
        self.model_config = self._get_model_config(model_key)

        self.analysis_results = []
        self.valid_packages = self._load_valid_packages()

        # Initialise model clients
        provider = self.model_config.get('provider')
        self.openai_client = OpenAI(api_key=openai_api_key) if provider == 'openai' and openai_api_key else None
        self.claude_client = Anthropic(api_key=claude_api_key) if provider == 'claude' and claude_api_key else None
        self.deepseek_client = (OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
                                if provider == 'deepseek' and deepseek_api_key else None)
        self.hf_client = InferenceClient(token=hf_token, model=self.model_config['hf_name']) \
            if provider not in ['openai', 'claude', 'deepseek'] and hf_token else None



    def _get_model_config(self, model_key):
        for section in ['commercial', 'open_source']:
            if model_key in MODELS_CONFIG.get(section, {}):
                return MODELS_CONFIG[section][model_key]
        raise ValueError(f"Model key '{model_key}' not found in MODELS_CONFIG")

    def _load_valid_packages(self):
        try:
            file_path = self.data_paths['packages']
            if not os.path.exists(file_path):
                print(f"Warning: Packages file not found: {file_path}")
                return set()

            df = pd.read_csv(file_path)
            if 'name' in df.columns:
                return set(df['name'].astype(str).str.lower())
            if 'package_name' in df.columns:
                return set(df['package_name'].astype(str).str.lower())
            return set(df.iloc[:, 0].astype(str).str.lower())
        except Exception as e:
            print(f"Error loading valid packages: {e}")
            return set()

    def is_hallucinated(self, package_name):
        return package_name.lower() not in self.valid_packages


    # Code generation

    def generate_code_from_prompt(self, prompt):
        provider = self.model_config['provider']
        model_name = self.model_config['api_name']
        system_msg = SYSTEM_MESSAGES[self.language]['code_generation']

        try:
            if provider == 'openai' and self.openai_client:
                resp = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": system_msg},
                              {"role": "user", "content": prompt}],
                    temperature=0, max_tokens=512
                )
                return resp.choices[0].message.content.strip()

            if provider == 'claude' and self.claude_client:
                resp = self.claude_client.messages.create(
                    model=model_name,
                    system=system_msg,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512
                )
                return resp.content[0].text.strip() if resp.content else ""

            if provider == 'deepseek' and self.deepseek_client:
                resp = self.deepseek_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": system_msg},
                              {"role": "user", "content": prompt}],
                    temperature=0, max_tokens=512
                )
                return resp.choices[0].message.content.strip()

            if self.hf_client:
                return self.hf_client.text_generation(prompt, max_new_tokens=512).strip()

        except Exception as e:
            print(f"Error generatin code : {e}")
        return "Code generation failed"


    # Heuristic 1
    def extract_packages_heuristic1(self, code):
        packages = set()
        if self.language == 'python':
            pip_patterns = [
                r'pip\s+install\s+([a-zA-Z0-9_\-\.]+)',
                r'pip3\s+install\s+([a-zA-Z0-9_\-\.]+)',
                r'python\s+-m\s+pip\s+install\s+([a-zA-Z0-9_\-\.]+)',
                r'python3\s+-m\s+pip\s+install\s+([a-zA-Z0-9_\-\.]+)'
            ]
            for p in pip_patterns:
                packages.update(re.findall(p, code, flags=re.IGNORECASE | re.MULTILINE))
            packages.update(re.findall(r'^\s*(?:import|from)\s+([a-zA-Z0-9_]+)', code, flags=re.MULTILINE))

        elif self.language == 'javascript':
            npm_patterns = [
                r'npm\s+install\s+([a-zA-Z0-9_\-@/\.]+)',
                r'npm\s+i\s+([a-zA-Z0-9_\-@/\.]+)',
                r'yarn\s+add\s+([a-zA-Z0-9_\-@/\.]+)',
                r'pnpm\s+install\s+([a-zA-Z0-9_\-@/\.]+)',
                r'pnpm\s+add\s+([a-zA-Z0-9_\-@/\.]+)'
            ]
            for p in npm_patterns:
                packages.update(re.findall(p, code, flags=re.IGNORECASE | re.MULTILINE))
            import_patterns = [
                r"require\(['\"]([^'\"]+)['\"]\)",
                r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",
                r"import\s+['\"]([^'\"]+)['\"]"
            ]
            
            for p in import_patterns:
                for match in re.findall(p, code):
                    if not match.startswith(('.', '/')):
                        pkg = match.split('/')[0]
                        if pkg.startswith('@') and '/' in match:
                            pkg = '/'.join(match.split('/')[:2])
                        packages.add(pkg)
        return list(packages)

    # Model query and parsing
    def _query_model(self, system_msg, content):
        provider = self.model_config['provider']
        model_name = self.model_config['api_name']

        try:
            if provider == 'openai' and self.openai_client:
                resp = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": system_msg},
                              {"role": "user", "content": content}],
                    temperature=0, max_tokens=512
                )
                return resp.choices[0].message.content

            if provider == 'claude' and self.claude_client:
                resp = self.claude_client.messages.create(
                    model=model_name,
                    system=system_msg,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=512
                )
                return resp.content[0].text if resp.content else ""

            if provider == 'deepseek' and self.deepseek_client:
                resp = self.deepseek_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": system_msg},
                              {"role": "user", "content": content}],
                    temperature=0, max_tokens=512
                )
                return resp.choices[0].message.content

            if self.hf_client:
                return self.hf_client.text_generation(f"{system_msg}\n{content}", max_new_tokens=256)

        except Exception as e:
            print(f"Error querying model: {e}")
        return ""

    def _parse_llm_response(self, response):
        if not response or response.strip().lower() == 'none':
            return []
        cleaned = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
        cleaned = re.sub(r'[^a-zA-Z0-9_,\-\s@/\.]', ' ', cleaned)
        return [re.sub(r'[^a-zA-Z0-9_\-@/\.]', '', part.strip())
                for part in cleaned.split(',') if part.strip() and len(part.strip()) > 1]

    # Heuristic 2 
    def query_model_for_packages_heuristic2(self, code_snippet):
        system_msg = SYSTEM_MESSAGES.get(self.language, {}).get(
            'package_from_code', f"List all packages required to run this {self.language} code:")
        return self._query_model(system_msg, code_snippet)
    # Heuristic 3 
    def query_model_for_packages_heuristic3(self, original_prompt):
        system_msg = SYSTEM_MESSAGES.get(self.language, {}).get(
            'package_from_prompt', f"List all {self.language} packages needed for this task:")
        return self._query_model(system_msg, original_prompt)

    def query_model_for_packages(self, code_snippet):
        return self.query_model_for_packages_heuristic2(code_snippet)


    # analysis
    def analyze_code_sample(self, code, original_prompt=None, model_name=None, is_commercial=None):
        h1 = self.extract_packages_heuristic1(code)
        h2 = self._parse_llm_response(self.query_model_for_packages_heuristic2(code))
        h3 = self._parse_llm_response(self.query_model_for_packages_heuristic3(original_prompt)) if original_prompt else []
        all_packages = list(set(h1 + h2 + h3))
        hallucinated = [pkg for pkg in all_packages if self.is_hallucinated(pkg)]

        result = {
            "language": self.language,
            "model": model_name or self.model_key,
            "total_packages": len(all_packages),
            "hallucinated_packages": len(hallucinated),
            "hallucinations": hallucinated,
            "packages_detected": all_packages,
            "package_breakdown": {
                "install_packages": all_packages,
                "import_packages": h1,
                "llm_suggested": h2 + h3,
                "heuristic1_packages": h1,
                "heuristic2_packages": h2,
                "heuristic3_packages": h3,
                "all_packages": all_packages
            }
        }
        self.analysis_results.append(result)
        return result


    # Stats
    def calculate_final_rates(self):
        detected, hallucinated, valid = set(), set(), set()
        for res in self.analysis_results:
            for pkg in res["packages_detected"]:
                (hallucinated if self.is_hallucinated(pkg) else valid).add(pkg)
                detected.add(pkg)
        return {
            "total_detected": len(detected),
            "total_hallucinated": len(hallucinated),
            "total_valid": len(valid),
            "hallucination_rate": len(hallucinated) / max(len(detected), 1),
            "all_hallucinated_packages": sorted(hallucinated),
            "all_valid_packages": sorted(valid)
        }

    def save_results_and_calculate_rates(self, output_dir="results", verbose=True):
        os.makedirs(output_dir, exist_ok=True)
        stats = self.calculate_final_rates()
        with open(os.path.join(output_dir, f"{self.language}_hallucinated_packages.txt"), 'w', encoding='utf-8') as f:
            f.write("\n".join(stats["all_hallucinated_packages"]))
        with open(os.path.join(output_dir, f"{self.language}_valid_detected_packages.txt"), 'w', encoding='utf-8') as f:
            f.write("\n".join(stats["all_valid_packages"]))
        with open(os.path.join(output_dir, f"{self.language}_final_report.txt"), 'w', encoding='utf-8') as f:
            f.write(f"FINAL REPORT - {self.language.upper()}\n{'=' * 40}\n")
            f.write(f"Total detected packages: {stats['total_detected']}\n")
            f.write(f"Hallucinated packages: {stats['total_hallucinated']}\n")
            f.write(f"Valid packages: {stats['total_valid']}\n")
            f.write(f"Hallucination rate: {stats['hallucination_rate']:.2%}\n")
        if verbose:
            print(f"Results saved in {output_dir}, hallucination rate {stats['hallucination_rate']:.2%}")
        return stats