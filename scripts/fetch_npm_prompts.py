import requests
import os
import json


output_path = "../data/javascript/master/top250-pkg_desc_pypi.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def fetch_npm_packages(query="prompt", size=250):
    url = f"https://api.npms.io/v2/search?q={query}&size={size}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['results']
    else:
        print(f"Erreur {response.status_code}: {response.text}")
        return []

def save_packages_to_jsonl(packages, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for pkg in packages:
            name = pkg['package']['name']
            desc = pkg['package'].get('description', '')
            line = {
                "package_name": name,
                "description": desc
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"File Doawnloaded : {filepath}")

packages = fetch_npm_packages(query="prompt", size=250)
save_packages_to_jsonl(packages, output_path)