from dotenv import load_dotenv
import os
import google.genai as genai

load_dotenv()
gemini_api_key=os.getenv("GEMINI_API_KEY")
client=genai.Client(api_key=gemini_api_key)

def askGemini(words):
    response=client.models.generate_content(
        model='gemini-2.0-flash',
        config=genai.types.GenerateContentConfig(
            system_instruction="Pretend you are a sign language interpreter. A sequence of signed words will be provided, and I want you to interpret them into natural, conversational English, while including commonly used phrases. Give me the sentence only"
        ),
        contents=words #assuming the input is a string
    )
    return response.text

