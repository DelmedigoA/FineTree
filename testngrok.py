import requests
import json

url = "https://084e-34-10-105-120.ngrok-free.app/v1/chat/completions"

payload = {
    "model": "Qwen/Qwen3.5-27B",
    "messages": [
        {"role": "user", "content": "Say hello in Hebrew."}
    ],
    "max_tokens": 20,
    "temperature": 0,
}
resp = requests.post(
    url,
    headers={
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "1"
    },
    json=payload,
    timeout=60
)

data = resp.json()

if "error" in data:
    print("ERROR:", json.dumps(data["error"], indent=2))
else:
    print(data["choices"][0]["message"]["content"])