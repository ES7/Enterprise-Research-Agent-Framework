import os
from openai import OpenAI
from dotenv import load_dotenv
from config import MODEL

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat(messages, tools=None, tool_choice="auto"):

    params = {
        "model": MODEL,
        "messages": messages
    }

    if tools is not None:
        params["tools"] = tools
        params["tool_choice"] = tool_choice

    return client.chat.completions.create(**params)