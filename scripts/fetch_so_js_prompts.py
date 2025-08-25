import requests, json, os

os.makedirs("../data/javascript/prompts", exist_ok=True)

def fetch_so_questions(tag, pagesize=5):
    url = (
        f"https://api.stackexchange.com/2.3/questions"
        f"?order=desc&sort=votes&tagged={tag}&site=stackoverflow"
        f"&pagesize={pagesize}&filter=withbody"
    )
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception(f"API request failed with status {r.status_code}")
    data = r.json()
    if "items" not in data:
        raise Exception("No 'items' in API response")
    return data

# Test with smaller size (max 100 allowed per API call)
data = fetch_so_questions("javascript", 100)

with open("../data/javascript/prompts/SO_QST.jsonl", "w", encoding="utf-8") as f:
    for i, q in enumerate(data["items"]):
        json.dump({
            "id": i,
            "Title": q.get("title", ""),
            "Body": q.get("body", "")
        }, f, ensure_ascii=False)
        f.write("\n")
