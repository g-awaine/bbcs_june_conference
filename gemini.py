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
           system_instruction = (
                "You are an ASL interpreter. You will be given a sequence of signed words. "
                "Your job is to interpret them into correct, natural-sounding English sentences. "
                "Do not add any extra information or explanation. "
                "Keep the output faithful to the meaning of the original words. "
                "Only return the corrected sentence."
            )
        ),
        contents=words #assuming the input is a string
    )
    return response.text

