import pandas as pd
import requests, os

os.makedirs("data/master", exist_ok=True)

# ready to use repo that scraped the most downloaded packages
url = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.csv"

df = pd.read_csv(url)

top5000 = df["project"].head(5000)

# save in data/master/top5000-pypi.txt
output_file = "data/master/top5000-pypi.csv"
top5000.to_csv(output_file, index=False, header=False)

