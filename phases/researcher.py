import json
from config import MIN_SEARCHES, MAX_SCHEMA_RETRIES, TOKEN_BUDGET
from core.metrics import count_tokens, estimate_cost
from tools.search import web_search
from llm.client import chat


RESEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "insights": {
            "type": "array",
            "items": {"type": "string"}
        },
        "pros": {
            "type": "array",
            "items": {"type": "string"}
        },
        "cons": {
            "type": "array",
            "items": {"type": "string"}
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["query", "insights", "pros", "cons", "sources"]
}



def extract_structured_insights(query, search_results):

    for attempt in range(MAX_SCHEMA_RETRIES):

        messages = [
            {
                "role": "system",
                "content": (
                    "You must return STRICT valid JSON only. "
                    "No markdown. No explanation. "
                    "Schema:\n"
                    "{\n"
                    '  "query": string,\n'
                    '  "insights": [string],\n'
                    '  "pros": [string],\n'
                    '  "cons": [string],\n'
                    '  "sources": [string]\n'
                    "}"
                )
            },
            {
                "role": "user",
                "content": json.dumps({
                    "query": query,
                    "search_results": search_results
                })
            }
        ]

        response = chat(messages)

        content = response.choices[0].message.content.strip()

        # Remove accidental markdown formatting
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(content)

            # Basic schema validation
            required_keys = ["query", "insights", "pros", "cons", "sources"]
            for key in required_keys:
                if key not in parsed:
                    raise ValueError("Missing key")

            return parsed

        except Exception as e:
            if attempt == MAX_SCHEMA_RETRIES - 1:
                # Fallback instead of crashing
                return {
                    "query": query,
                    "insights": ["Extraction failed"],
                    "pros": [],
                    "cons": [],
                    "sources": []
                }


def research_phase(state):

    while state.search_count < MIN_SEARCHES:

        state.iterations += 1

        query_prompt = [
            {"role": "system", "content": "Generate a precise research query."},
            {"role": "user", "content": state.goal}
        ]

        token_count = count_tokens(query_prompt)
        state.total_tokens += token_count

        if state.total_tokens > TOKEN_BUDGET:
            state.log("TERMINATED → Token budget exceeded")
            break

        response = chat(query_prompt)
        query = response.choices[0].message.content.strip()

        # Convergence detection
        if query in state.query_history:
            state.log("TERMINATED → Duplicate query detected")
            break

        state.add_query(query)
        state.log(f"SEARCH QUERY → {query}")

        results = web_search(query)
        state.search_count += 1

        structured = extract_structured_insights(query, results)

        # Confidence score (simple heuristic)
        structured["confidence"] = min(
            1.0,
            0.5 + (len(structured.get("insights", [])) * 0.1)
        )

        state.add_research_item(structured)
        state.log(f"STRUCTURED ITEM STORED → {query}")

    state.estimated_cost = estimate_cost(state.total_tokens)