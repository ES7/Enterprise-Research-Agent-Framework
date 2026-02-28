from llm.client import chat
import json


def synthesize_report(state):

    messages = [
        {
            "role": "system",
            "content": (
                "Generate a professional enterprise research report. "
                "Use ONLY the structured research data provided. "
                "Do NOT fabricate information."
            )
        },
        {
            "role": "user",
            "content": json.dumps(state.research_items)
        }
    ]

    response = chat(messages)

    return response.choices[0].message.content