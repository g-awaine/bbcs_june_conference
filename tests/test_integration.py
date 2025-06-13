# Run the tts.py Flask API service first, then use this script to test.
import requests
import sys
import time
import os
# Add parent directory to sys.path so we can import gemini.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gemini import askGemini
# Testing without SgSL portion for now

LIST_OF_WORDS = ["I","eat","I", "I", "breakfast","breakfast", "satay"]

# Now define the TTS service
def test_tts(gemini_response):
    url = "http://localhost:5001/speak"
    payload = {"text": gemini_response}
    headers = {"Content-Type": "application/json"}

    print(f"Sending TTS request with text: {gemini_response}")
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        print("TTS service responded successfully.")
        print("Response:", response.json())
    else:
        print(f"TTS service failed with status code: {response.status_code}")
        print("Response:", response.text)

# Call Gemini
def test_gemini():
    words = " ".join(LIST_OF_WORDS)
    print(f"Testing Gemini with words: {words}")
    response = askGemini(words)
    print(f"Gemini response: {response}")
    return response

gemini_response = test_gemini()
if not gemini_response:
    print("Gemini response is empty. Exiting test.")
    sys.exit(1)
else:
    print("Gemini response received successfully.")
    test_tts(gemini_response)

