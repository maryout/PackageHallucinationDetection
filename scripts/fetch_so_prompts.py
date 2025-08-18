import requests, json, os

os.makedirs("data/prompts", exist_ok=True)

def fetch_so_questions(tag, pagesize=5):
    url = f"https://api.stackexchange.com/2.3/questions?order=desc&sort=votes&tagged={tag}&site=stackoverflow&pagesize={pagesize}"
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception(f"API request failed with status {r.status_code}")
    data = r.json()
    if "items" not in data:
        raise Exception("No 'items' in API response")
    return data

# Test1 for 1000 rows Python StackOverflow prompts -- faild only 100 are alowed
data = fetch_so_questions("python", 100)

with open("data/prompts/so-py-sample.jsonl", "w", encoding="utf-8") as f:
    for i, q in enumerate(data["items"]):
        json.dump({
            "id": i,
            "language": "python",
            "source": "stack_overflow",
            "prompt": q.get("title", "")
        }, f)
        f.write("\n")
