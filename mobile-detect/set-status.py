# Sæt status på hjemmesiden manuelt – nyttigt til at teste serveren uden webcam.
# Brug:  uv run mobile-detect/set-status.py true   (rød: mobil fundet)
#        uv run mobile-detect/set-status.py false  (hvid: alt OK)
import sys
import requests

API_URL = "http://localhost:8080/update"

status = len(sys.argv) > 1 and sys.argv[1].lower() == "true"
requests.post(API_URL, json={"status": status}, timeout=2)
print("🔴 Status: mobil fundet" if status else "⚪ Status: alt OK")
