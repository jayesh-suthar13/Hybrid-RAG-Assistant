import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("Error: GROQ_API_KEY missing!")
else:
    print(f"API Key found: {api_key[:8]}...")
    # Quick connectivity test
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)
    res = llm.invoke("Say connection success")
    print("LLM Response:", res.content)