import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent import create_agent_graph, get_dataframe_schema, get_system_prompt

st.set_page_config(layout="wide")
st.markdown(
    """
    <h1 style='text-align: center;'>DataBot</h1>
    <h4 style='text-align: center; color: gray;'>AI Agent for Data Science</h4>
""",
    unsafe_allow_html=True,
)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("dataframe", None)
st.session_state.setdefault("agent", None)
st.session_state.setdefault("current_file_name", None)


def initialize_agent(df: pd.DataFrame, file_name: str):
    st.session_state.messages.clear()
    st.session_state.dataframe = df
    try:
        st.session_state.agent = create_agent_graph(df)
        schema = get_dataframe_schema(df)
        prompt = get_system_prompt(schema)
        st.session_state.messages.append(SystemMessage(content=prompt))
        st.session_state.current_file_name = file_name
    except Exception as e:
        st.error(f"Agent initialization failed: {e}")
        st.session_state.agent = None
        st.session_state.dataframe = None
        st.session_state.current_file_name = None


with st.sidebar:
    st.header("Upload your CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        if (
            st.session_state.current_file_name != uploaded_file.name
            or st.session_state.agent is None
        ):
            try:
                df = pd.read_csv(uploaded_file)
                initialize_agent(df, uploaded_file.name)
            except Exception as e:
                st.error(f"Error reading or processing CSV: {e}")
                st.session_state.dataframe = None
                st.session_state.agent = None
                st.session_state.current_file_name = None
                st.session_state.messages.clear()

        if st.session_state.dataframe is not None:
            st.success("CSV successfully uploaded.")
            st.header("Data Preview")
            st.dataframe(st.session_state.dataframe.head(), height=200)
            if st.session_state.agent:
                st.success("Agent is ready.")
                st.info("You can now chat about your data.")
            else:
                st.warning("Agent initialization failed.")
    else:
        if st.session_state.current_file_name is not None:
            st.session_state.messages.clear()
            st.session_state.dataframe = None
            st.session_state.agent = None
            st.session_state.current_file_name = None
            st.info("File removed. Agent and chat history cleared.")


for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        st.chat_message("user").write(message.content)
    elif isinstance(message, AIMessage) and message.content:
        st.chat_message("assistant").write(message.content)

if user_query := st.chat_input("Ask something about your data..."):
    if st.session_state.dataframe is None or st.session_state.agent is None:
        st.warning("Please upload and initialize a CSV file first.")
    else:
        st.session_state.messages.append(HumanMessage(content=user_query))
        st.chat_message("user").write(user_query)

        messages_for_agent = list(st.session_state.messages)

        with st.spinner("Thinking..."):
            try:
                final_ai_message = None
                for event in st.session_state.agent.stream(
                    {"messages": messages_for_agent},
                    stream_mode="values",
                ):
                    if event["messages"]:
                        final_ai_message = event["messages"][-1]

                if isinstance(final_ai_message, AIMessage) and final_ai_message.content:
                    st.session_state.messages.append(final_ai_message)
                    st.chat_message("assistant").write(final_ai_message.content)
                elif final_ai_message is None:
                    st.warning("The agent did not return a response.")

            except Exception as e:
                st.error(f"Error processing your query: {e}")