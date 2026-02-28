import streamlit as st
from agent import run_agent

st.set_page_config(page_title="Research Agent", layout="wide")

st.title("Autonomous Research Agent")

topic = st.text_input("Enter Research Topic")

if st.button("Run Agent") and topic:

    with st.spinner("Running agent..."):
        result = run_agent(topic)

    st.subheader("Research Report")
    st.write(result["report"])

    st.subheader("Execution Logs")
    for log in result["logs"]:
        st.write(log)

    st.subheader("Steps Used")
    st.subheader("Search Calls")
    st.write(result["searches"])