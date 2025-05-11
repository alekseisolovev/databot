import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

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


def initialize_agent(df: pd.DataFrame):
    st.session_state.dataframe = df
    st.session_state.messages.clear()
    if df is not None:
        try:
            st.session_state.agent = create_agent_graph(df)
            schema = get_dataframe_schema(df)
            prompt = get_system_prompt(schema)
            st.session_state.messages.append(SystemMessage(content=prompt))
        except Exception as e:
            st.error(f"Agent initialization failed: {e}")
            st.session_state.agent = None
            st.session_state.dataframe = None
    else:
        st.session_state.agent = None


with st.sidebar:
    st.header("Upload your CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            initialize_agent(df)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            initialize_agent(None)

    if st.session_state.dataframe is not None:
        st.success("CSV successfully uploaded.")
        st.header("Data Preview")
        st.dataframe(st.session_state.dataframe.head(), height=200)
        if st.session_state.agent:
            st.success("Agent initialized.")
            st.info("You can now chat about your data.")
        else:
            st.warning("Agent initialization failed.")

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        st.chat_message("user").write(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("assistant").write(message.content)

if user_query := st.chat_input("Ask something about your data..."):
    if st.session_state.dataframe is None or st.session_state.agent is None:
        st.warning(
            "Please upload a CSV file first and ensure the agent is initialized."
        )
    else:
        st.session_state.messages.append(HumanMessage(content=user_query))
        st.chat_message("user").write(user_query)

        messages_for_agent = list(st.session_state.messages)

        with st.spinner("Thinking..."):
            try:
                final_messages = None
                for event in st.session_state.agent.stream(
                    {"messages": messages_for_agent},
                    stream_mode="values",
                ):
                    final_messages = event["messages"]

                if final_messages:
                    for message in final_messages:
                        if (
                            isinstance(message, AIMessage)
                            and message not in st.session_state.messages
                        ):
                            if message.content:
                                st.session_state.messages.append(message)
                                st.chat_message("assistant").write(message.content)
                                break
                else:
                    error_message = AIMessage(
                        content="Agent did not return a response."
                    )
                    st.session_state.messages.append(error_message)
                    st.chat_message("assistant").write(error_message.content)

            except Exception as e:
                error_message = f"Error processing your query: {e}"
                st.error(error_message)
                error_message = AIMessage(content=error_message)
                st.session_state.messages.append(error_message)
                st.chat_message("assistant").write(error_message.content)
                