import pandas as pd
import requests
import json
import os
import time
from typing import Optional

# Create folder data/master/
os.makedirs("data/master", exist_ok=True)

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

# From Ready to use repo
url = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.csv"
df = pd.read_csv(url)

# Extraction top 5000 fior the first try
top5000 = df["project"].head(5000)

output_file = "data/master/top5-pkg_desc_pypi.jsonl"

with open(output_file, 'w', encoding='utf-8') as f:
    for i, package_name in enumerate(top5000):
        print(f"Processing {i+1}/5000: {package_name}")
        
        # Save the description 
        description = get_package_description(package_name)
        
        # Create JSONl file
        package_data = {
            "package_name": package_name,
            "description": description if description else ""
        }
        
        f.write(json.dumps(package_data, ensure_ascii=False) + '\n')
        time.sleep(0.1)