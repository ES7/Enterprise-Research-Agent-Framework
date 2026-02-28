import streamlit as st
from core.controller import run_agent

st.set_page_config(page_title="AI Research Framework", layout="wide")

st.title("Enterprise Research Agent Framework")

goal = st.text_input("Enter Research Topic")

if st.button("Run Agent") and goal:

    with st.spinner("Running structured research pipeline..."):
        result = run_agent(goal)

    st.subheader("Final Report")
    st.write(result["report"])

    st.subheader("Execution Logs")
    for log in result["logs"]:
        st.write(log)

    st.subheader("Search Count")
    st.write(result["search_count"])

    st.subheader("Iterations")
    st.write(result["iterations"])

    st.subheader("Structured Research Data")
    st.json(result["structured_data"])
    
    st.subheader("Tokens Used")
    st.write(result["tokens_used"])

    st.subheader("Estimated Cost (USD)")
    st.write(result["estimated_cost"])

    st.subheader("Run ID")
    st.write(result["run_id"])