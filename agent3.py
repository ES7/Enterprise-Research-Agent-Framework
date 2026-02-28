import os
import json
from dotenv import load_dotenv
from ddgs import DDGS
from openai import OpenAI
import tiktoken

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"
MAX_ITERATIONS = 8
TOKEN_BUDGET = 10000
MIN_RESEARCH_CALLS = 3


# =========================
# STATE
# =========================

class AgentState:
    def __init__(self, goal):
        self.goal = goal
        self.step_count = 0
        self.search_count = 0
        self.notes = []
        self.total_tokens = 0
        self.finished = False


# =========================
# TOKEN COUNT
# =========================

encoding = tiktoken.encoding_for_model(MODEL)

def count_tokens(messages):
    total = 0
    for msg in messages:
        if isinstance(msg.get("content"), str):
            total += len(encoding.encode(msg["content"]))
    return total


# =========================
# TOOLS
# =========================

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


# =========================
# LLM CALL
# =========================

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


# =========================
# PHASE 1 — RESEARCH LOOP
# =========================

def research_phase(state, logs):

    messages = [
        {
            "role": "system",
            "content": (
                "You are in research phase. "
                "You MUST call web_search at least 3 times. "
                "After each search, extract 3–5 key insights "
                "and store them in structured bullet points."
            )
        },
        {"role": "user", "content": state.goal}
    ]

    while state.search_count < MIN_RESEARCH_CALLS and state.step_count < MAX_ITERATIONS:

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

                logs.append(f"RESEARCH TOOL → {tool_name}: {arguments}")

                result = execute_tool(tool_name, arguments)
                state.search_count += 1

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

                # Extract structured notes immediately
                note_prompt = [
                    {
                        "role": "system",
                        "content": "Extract 5 concise structured bullet insights."
                    },
                    {
                        "role": "user",
                        "content": json.dumps(result)
                    }
                ]

                note_response = client.chat.completions.create(
                    model=MODEL,
                    messages=note_prompt
                )

                extracted = note_response.choices[0].message.content
                state.notes.append(extracted)

        else:
            break


# =========================
# PHASE 2 — REPORT GENERATION
# =========================

def report_phase(state):

    report_prompt = [
        {
            "role": "system",
            "content": (
                "Generate a professional research report "
                "using ONLY the structured notes provided. "
                "Do not fabricate new information."
            )
        },
        {
            "role": "user",
            "content": json.dumps(state.notes)
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=report_prompt
    )

    return response.choices[0].message.content


# =========================
# MAIN AGENT
# =========================

def run_agent(goal):

    state = AgentState(goal)
    logs = []

    research_phase(state, logs)

    report = report_phase(state)

    return {
        "report": report,
        "logs": logs,
        "steps": state.step_count,
        "searches": state.search_count
    }