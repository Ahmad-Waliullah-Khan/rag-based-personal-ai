# cli_friday.py
from app_logic import query_ai

while True:
    query = input("🧑 You: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = query_ai(query)
    print(f"🤖 {response}")
