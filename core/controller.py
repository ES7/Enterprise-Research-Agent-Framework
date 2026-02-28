from core.state import ResearchState
from phases.researcher import research_phase
from phases.synthesizer import synthesize_report


def run_agent(goal):

    state = ResearchState(goal)

    research_phase(state)

    if state.search_count == 0:
        state.log("FAILED → No research collected")
        return {
            "error": "Research phase failed",
            "logs": state.logs
        }

    report = synthesize_report(state)

    state.finished = True

    return {
        "run_id": state.run_id,
        "report": report,
        "logs": state.logs,
        "search_count": state.search_count,
        "iterations": state.iterations,
        "structured_data": state.research_items,
        "tokens_used": state.total_tokens,
        "estimated_cost": round(state.estimated_cost, 6)
    }