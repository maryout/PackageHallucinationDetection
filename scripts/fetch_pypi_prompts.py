import pandas as pd
import requests
import json
import os
import time
from typing import Optional


os.makedirs("../data/pythonN/master", exist_ok=True)

def get_package_description(package_name: str) -> Optional[str]:
    """Fetch package description from PyPI API"""
    try:
        url = f"https://pypi.org/pypi/{package_name}/json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("info", {}).get("summary", "")
        return None
    except Exception as e:
        print(f"Error fetching {package_name}: {e}")
        return None

url = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.csv"
df = pd.read_csv(url)

# for test to changer befor commit
top5000 = df["project"].head(5)

jsonl_file = "../data/master/top5-pkg_desc_pypi.jsonl"
csv_file = "../data/master/top5-pkg_desc_pypi.csv"

records = []

with open(jsonl_file, 'w', encoding='utf-8') as f:
    for i, package_name in enumerate(top5000):
        print(f"Processing {i+1}/{len(top5000)}: {package_name}")
        
        description = get_package_description(package_name)
        
        package_data = {
            "package_name": package_name,
            "description": description if description else ""
        }
        
        f.write(json.dumps(package_data, ensure_ascii=False) + '\n')
        
        records.append(package_name)
        
        time.sleep(0.1)

df_out = pd.DataFrame(records)
df_out.to_csv(csv_file, index=False, encoding="utf-8", header=False)

print(f"Done! Saved {len(records)} packages to:\n- {jsonl_file}\n- {csv_file}")
