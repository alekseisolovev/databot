import logging

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from agent import Agent

logger = logging.getLogger(__name__)


plt.style.use("dark_background")

st.markdown(
    """
    <h1 style='text-align: center;'>DataBot</h1>
    <h4 style='text-align: center; color: gray;'>AI Agent for Data Science</h4>
""",
    unsafe_allow_html=True,
)

st.session_state.setdefault("dataframe", None)
st.session_state.setdefault("agent", None)
st.session_state.setdefault("current_file_name", None)


def reset_session_state():
    st.session_state.dataframe = None
    st.session_state.agent = None
    st.session_state.current_file_name = None


def initialize_agent(df: pd.DataFrame, file_name: str):
    logger.info(f"Agent: Attempting initialization for file '{file_name}'.")
    try:
        st.session_state.agent = Agent(df)
        st.session_state.current_file_name = file_name
        logger.info(f"Agent: Initialization successful for file '{file_name}'.")
    except Exception as e:
        logger.error(
            f"Agent: Initialization failed for file '{file_name}'. Error: {e}",
            exc_info=True,
        )
        st.error(f"Agent initialization failed. Error: {e}")
        reset_session_state()


with st.sidebar:
    st.header("Upload your CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        if (
            st.session_state.current_file_name != uploaded_file.name
            or st.session_state.agent is None
        ):
            logger.info(f"File Uploader: New file '{uploaded_file.name}' selected.")
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.dataframe = df
                logger.info(
                    f"File Uploader: CSV '{uploaded_file.name}' read successfully. Shape: {df.shape}"
                )
                initialize_agent(df, uploaded_file.name)
            except Exception as e:
                logger.error(
                    f"File Uploader: Reading or processing CSV '{uploaded_file.name}' failed. Error: {e}",
                    exc_info=True,
                )
                st.error(f"Reading or processing CSV failed. Error: {e}")
                reset_session_state()

        if (
            st.session_state.dataframe is not None
            and st.session_state.current_file_name == uploaded_file.name
        ):
            st.success(f"CSV uploaded successfully.")
            st.header("Data Preview")
            st.dataframe(st.session_state.dataframe.head(), height=200)
            if st.session_state.agent:
                st.success(f"Agent is ready.")
            else:
                logger.warning(
                    f"File Uploader: DataFrame '{st.session_state.current_file_name}' is loaded, but agent is not ready."
                )
                st.warning(
                    f"Agent is not initialized. Check for errors above or try re-uploading."
                )
    else:
        if st.session_state.current_file_name is not None:
            previous_file_name = st.session_state.current_file_name
            logger.info(f"File Uploader: File '{previous_file_name}' removed by user.")
            reset_session_state()
            st.info(f"File '{previous_file_name}' removed. Agent and data cleared.")


if st.session_state.agent:
    for message in st.session_state.agent.get_messages():
        if isinstance(message, (SystemMessage, ToolMessage)):
            continue

        if isinstance(message, AIMessage):
            is_tool_call = bool(message.tool_calls)
            has_content = bool(message.content.strip())
            has_artifacts = (
                "dataframe_artifact" in message.additional_kwargs
                or "figure_artifact" in message.additional_kwargs
            )

            if is_tool_call and not has_content and not has_artifacts:
                continue

            if not is_tool_call and not has_content and not has_artifacts:
                with st.chat_message(message.type):
                    st.warning("Agent provided no response or artifact for this turn.")
                continue

        with st.chat_message(message.type):
            if message.content.strip():
                st.write(message.content)

            if isinstance(message, AIMessage):
                if "dataframe_artifact" in message.additional_kwargs:
                    dataframe_artifact = message.additional_kwargs["dataframe_artifact"]
                    if isinstance(dataframe_artifact, (pd.DataFrame, pd.Series)):
                        st.dataframe(dataframe_artifact)
                if "figure_artifact" in message.additional_kwargs:
                    figure_artifact = message.additional_kwargs["figure_artifact"]
                    if isinstance(figure_artifact, matplotlib.figure.Figure):
                        st.pyplot(figure_artifact)


if user_query := st.chat_input("Ask something about your data..."):
    logger.info(f"Chat Input: User query received: '{user_query}'")
    if st.session_state.dataframe is None or st.session_state.agent is None:
        logger.warning(
            "Chat Input: Query attempt failed. Data or agent not initialized."
        )
        st.warning(
            "Please upload a CSV file and ensure the agent is initialized before asking questions."
        )
    else:
        st.chat_message("user").write(user_query)

        with st.spinner("Thinking..."):
            try:
                st.session_state.agent.invoke(user_query)

                ai_message = st.session_state.agent.get_messages()[-1]
                if isinstance(ai_message, AIMessage):
                    if ai_message.content:
                        logger.info(
                            f"Chat Response: Agent content received: '{str(ai_message.content)}'"
                        )
                    if "dataframe_artifact" in ai_message.additional_kwargs:
                        dataframe_artifact = ai_message.additional_kwargs[
                            "dataframe_artifact"
                        ]
                        logger.info(
                            f"Chat Response: Agent artifact received. Type: {type(dataframe_artifact)}, Shape: {getattr(dataframe_artifact, 'shape', 'N/A')}"
                        )
                    if "figure_artifact" in ai_message.additional_kwargs:
                        figure_artifact = ai_message.additional_kwargs[
                            "figure_artifact"
                        ]
                        logger.info(
                            f"Chat Response: Agent artifact received. Type: {type(figure_artifact)}"
                        )
                st.rerun()

            except Exception as e:
                logger.error(
                    f"Chat Response: Processing user query '{user_query}' failed. Error: {e}",
                    exc_info=True,
                )
                st.error(f"Processing your query '{user_query}' failed. Error: {e}")
