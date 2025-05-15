import logging

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent import create_agent_graph, get_dataframe_schema, get_system_prompt

logger = logging.getLogger(__name__)

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
    logger.info(f"Attempting to initialize agent for {file_name}")
    st.session_state.messages.clear()
    try:
        st.session_state.agent = create_agent_graph(df)
        schema = get_dataframe_schema(df)
        prompt = get_system_prompt(schema)
        st.session_state.messages.append(SystemMessage(content=prompt))
        st.session_state.current_file_name = file_name
        logger.info(f"Agent initialized successfully for {file_name}")
    except Exception as e:
        logger.error(f"Agent initialization failed for {file_name}: {e}", exc_info=True)
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
            logger.info(f"File uploaded: {uploaded_file.name}")
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.dataframe = df
                logger.info(
                    f"Successfully read CSV: {uploaded_file.name}. Shape: {df.shape}"
                )
                initialize_agent(df, uploaded_file.name)
            except Exception as e:
                logger.error(
                    f"Error reading CSV {uploaded_file.name}: {e}", exc_info=True
                )
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
                logger.error("Agent initialization failed.")
                st.warning("Agent initialization failed.")
    else:
        if st.session_state.current_file_name is not None:
            logger.info(f"File '{st.session_state.current_file_name}' removed.")
            st.session_state.messages.clear()
            st.session_state.dataframe = None
            st.session_state.agent = None
            st.session_state.current_file_name = None
            st.info("File removed. Agent and chat history cleared.")


for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        st.chat_message("user").write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            if message.content:
                st.write(message.content)
            if "dataframe" in message.additional_kwargs:
                artifact = message.additional_kwargs["dataframe"]
                if isinstance(artifact, (pd.DataFrame, pd.Series)):
                    st.dataframe(artifact, height=200)

if user_query := st.chat_input("Ask something about your data..."):
    logger.info(f"User query: '{user_query}'")
    if st.session_state.dataframe is None or st.session_state.agent is None:
        logger.warning("User attempted query without data or agent initialized.")
        st.warning("Please upload and initialize a CSV file first.")
    else:
        st.session_state.messages.append(HumanMessage(content=user_query))
        st.chat_message("user").write(user_query)

        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.invoke(
                    {"messages": st.session_state.messages}
                )
                ai_message = response["messages"][-1] if response["messages"] else None

                if isinstance(ai_message, AIMessage):
                    st.session_state.messages.append(ai_message)
                    with st.chat_message("assistant"):
                        if ai_message.content:
                            st.write(ai_message.content)
                            logger.info(
                                f"Agent response (content): {ai_message.content}"
                            )
                        if "dataframe" in ai_message.additional_kwargs:
                            artifact = ai_message.additional_kwargs["dataframe"]
                            if isinstance(artifact, (pd.DataFrame, pd.Series)):
                                st.dataframe(artifact)
                                logger.info(
                                    f"Agent response (artifact). Type: {type(artifact)}, Shape: {artifact.shape}"
                                )
                else:
                    logger.warning(
                        f"The agent did not return a valid AIMessage. Received: {ai_message}"
                    )
                    st.warning(
                        "The agent did not return a response or returned an unexpected format."
                    )

            except Exception as e:
                logger.error(
                    f"Error processing user query '{user_query}': {e}", exc_info=True
                )
                st.error(f"Error processing your query: {e}")
                