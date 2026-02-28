import os
import json
from dotenv import load_dotenv
from ddgs import DDGS
from openai import OpenAI
import tiktoken

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"
MAX_ITERATIONS = 6
TOKEN_BUDGET = 8000


class AgentState:
    def __init__(self, goal):
        self.goal = goal
        self.step_count = 0
        self.search_history = []
        self.total_tokens = 0
        self.finished = False


encoding = tiktoken.encoding_for_model(MODEL)


def count_tokens(messages):
    total = 0
    for msg in messages:
        if isinstance(msg.get("content"), str):
            total += len(encoding.encode(msg["content"]))
    return total


def web_search(query):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append({
                "title": r["title"],
                "body": r["body"],
                "href": r["href"]
            })
    return results


TOOLS = {
    "web_search": web_search
}


def validate_tool_call(name, arguments):
    if name not in TOOLS:
        raise ValueError("Tool does not exist.")

    if name == "web_search":
        if "query" not in arguments:
            raise ValueError("Missing query.")
        if not isinstance(arguments["query"], str):
            raise ValueError("Query must be string.")


def execute_tool(name, arguments):
    validate_tool_call(name, arguments)
    return TOOLS[name](**arguments)


def call_llm(messages):
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for research information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ],
        tool_choice="auto"
    )


def run_agent(goal):
    state = AgentState(goal)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research agent. "
                "Use web_search when needed. "
                "Collect insights before giving final answer. "
                "When finished, provide a structured research report."
            )
        },
        {"role": "user", "content": goal}
    ]

    logs = []

    while state.step_count < MAX_ITERATIONS:

        state.step_count += 1

        if count_tokens(messages) > TOKEN_BUDGET:
            logs.append("Token budget exceeded.")
            break

        response = call_llm(messages)
        message = response.choices[0].message

        if message.tool_calls:

            messages.append({
                "role": "assistant",
                "tool_calls": message.tool_calls,
                "content": None
            })

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                logs.append(f"TOOL CALL → {tool_name}: {arguments}")

                result = execute_tool(tool_name, arguments)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

        else:
            state.finished = True
            return {
                "report": message.content,
                "logs": logs,
                "steps": state.step_count
            }

    return {
        "report": "Stopped: Max iterations reached.",
        "logs": logs,
        "steps": state.step_count
    }