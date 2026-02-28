# Enterprise Research Agent Framework

Production-style, modular AI research framework built from first principles.

This is a state-driven, phase-controlled research execution engine, not a chatbot.

## Core Capabilities

- Structured memory (JSON state)
- Schema-enforced extraction
- Token budget enforcement
- Convergence and loop protection
- Deterministic orchestration
- Cost estimation
- Execution trace logging

## Architecture

### Phase 1: Structured Research Collection
- Generate focused research queries
- Run web search (DuckDuckGo)
- Extract strict JSON insights
- Validate schema and retry malformed outputs
- Detect duplicate queries
- Track tokens, budget, and confidence

### Phase 2: Report Synthesis
- Uses only structured research state
- No direct tool access
- No additional search
- Produces final enterprise-style report from collected evidence

## Structured State Shape

```json
{
  "query": "...",
  "insights": [...],
  "pros": [...],
  "cons": [...],
  "sources": [...],
  "confidence": 0.8
}
```

## Project Structure

```text
AI-Research-Agent/
|-- app.py
|-- config.py
|-- requirements.txt
|-- .env
|-- core/
|   |-- state.py
|   |-- controller.py
|   `-- metrics.py
|-- llm/
|   `-- client.py
|-- tools/
|   `-- search.py
`-- phases/
    |-- researcher.py
    `-- synthesizer.py
```

## Installation

```bash
git clone <your-repo-url>
cd AI-Research-Agent
python -m venv .venv
```

Activate virtual environment:
- Windows: `.venv\Scripts\activate`
- Mac/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env`:

```env
OPENAI_API_KEY=your_api_key_here
```

## Run

```bash
streamlit run app.py
```

## Runtime Output

- Final research report
- Execution logs
- Search and iteration counts
- Tokens used
- Estimated cost
- Run ID
- Structured research data
