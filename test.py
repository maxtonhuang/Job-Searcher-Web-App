import requests, json

URL = "https://api.mycareersfuture.gov.sg/v2/search"
params = {"limit": 3, "page": 0}
payload = {"search": "software engineer", "sessionId": ""}
headers = {"Content-Type": "application/json", "Accept": "application/json"}

resp = requests.post(URL, params=params, json=payload, headers=headers, timeout=20)
print("status:", resp.status_code)
data = resp.json()

# Show the top-level keys and the keys of the first result so we can map fields
print("top-level keys:", list(data.keys()))
results = data.get("results", data)
first = results[0] if isinstance(results, list) else results
print(json.dumps(first, indent=2)[:2000])
print("first result keys:", list(first.keys()))