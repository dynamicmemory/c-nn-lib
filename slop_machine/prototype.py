import requests 
import json 

url = "http://localhost:11434/api/generate"

while True: 
    prompt = input("Slop Machine >> ")
    
    if prompt == "quit":
        break

    payload = {
        "model": "gemma4:latest",
        "prompt": prompt,
        "stream": True 
        }
    
    # Send request
    res = requests.post(url, json=payload, stream=True)

    # Stream output
    for line in res.iter_lines():
        if line:
            data = json.loads(line)
    
            if "response" in data:
                print(data["response"], end="", flush=True)

    print()
