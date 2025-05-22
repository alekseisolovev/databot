import logging

import matplotlib.figure
import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent import create_agent_graph, get_dataframe_schema, get_system_prompt

logger = logging.getLogger(__name__)

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
    logger.info(f"Agent: Attempting initialization for file '{file_name}'.")
    st.session_state.messages.clear()
    try:
        st.session_state.agent = create_agent_graph(df)
        dataframe_schema = get_dataframe_schema(df)
        system_prompt = get_system_prompt(dataframe_schema)
        st.session_state.messages.append(SystemMessage(content=system_prompt))
        st.session_state.current_file_name = file_name
        logger.info(f"Agent: Initialization successful for file '{file_name}'.")
    except Exception as e:
        logger.error(
            f"Agent: Initialization failed for file '{file_name}'. Error: {e}",
            exc_info=True,
        )
        st.error(f"Agent initialization for '{file_name}' failed. Error: {e}")
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
                st.error(
                    f"Reading or processing CSV '{uploaded_file.name}' failed. Error: {e}"
                )
                st.session_state.dataframe = None
                st.session_state.agent = None
                st.session_state.current_file_name = None
                st.session_state.messages.clear()

        if (
            st.session_state.dataframe is not None
            and st.session_state.current_file_name == uploaded_file.name
        ):
            st.success(
                f"CSV '{st.session_state.current_file_name}' uploaded successfully."
            )
            st.header("Data Preview")
            st.dataframe(st.session_state.dataframe.head(), height=200)
            if st.session_state.agent:
                st.success(
                    f"Agent for '{st.session_state.current_file_name}' is ready."
                )
                st.info(
                    f"You can now chat about '{st.session_state.current_file_name}'."
                )
            else:
                logger.warning(
                    f"File Uploader: DataFrame '{st.session_state.current_file_name}' is loaded, but agent is not ready."
                )
                st.warning(
                    f"Agent for '{st.session_state.current_file_name}' is not initialized. Check for errors above or try re-uploading."
                )
    else:
        if st.session_state.current_file_name is not None:
            logger.info(
                f"File Uploader: File '{st.session_state.current_file_name}' removed by user."
            )
            st.session_state.messages.clear()
            st.session_state.dataframe = None
            st.session_state.agent = None
            previous_file_name = st.session_state.current_file_name
            st.session_state.current_file_name = None
            st.info(
                f"File '{previous_file_name}' removed. Agent and chat history cleared."
            )


for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue
    with st.chat_message(message.type):
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
        st.session_state.messages.append(HumanMessage(content=user_query))
        st.chat_message("user").write(user_query)
        with st.container():
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.invoke(
                        {"messages": st.session_state.messages}
                    )
                    ai_message = (
                        response["messages"][-1]
                        if response and response["messages"]
                        else None
                    )
                    if isinstance(ai_message, AIMessage):
                        st.session_state.messages.append(ai_message)
                        with st.chat_message("assistant"):
                            if ai_message.content:
                                st.write(ai_message.content)
                                logger.info(
                                    f"Chat Response: Agent content received: '{str(ai_message.content)}'"
                                )
                            if "dataframe_artifact" in ai_message.additional_kwargs:
                                dataframe_artifact = ai_message.additional_kwargs[
                                    "dataframe_artifact"
                                ]
                                if isinstance(
                                    dataframe_artifact, (pd.DataFrame, pd.Series)
                                ):
                                    st.dataframe(dataframe_artifact)
                                    logger.info(
                                        f"Chat Response: Agent artifact received. Type: {type(dataframe_artifact)}, Shape: {getattr(dataframe_artifact, 'shape', 'N/A')}"
                                    )
                            if "figure_artifact" in ai_message.additional_kwargs:
                                figure_artifact = ai_message.additional_kwargs[
                                    "figure_artifact"
                                ]
                                if isinstance(
                                    figure_artifact, matplotlib.figure.Figure
                                ):
                                    st.pyplot(figure_artifact)
                                    logger.info(
                                        f"Chat Response: Agent artifact received. Type: {type(figure_artifact)}"
                                    )
                            elif not ai_message.content:
                                logger.info(
                                    "Chat Response: AIMessage has no response or artifact for this turn."
                                )
                                st.warning(
                                    "Agent provided no response or artifact for this turn."
                                )
                    else:
                        logger.warning(
                            f"Chat Response: Agent did not return a valid AIMessage. Last message in response: {type(ai_message)}"
                        )
                        st.warning(
                            "Agent response issue: The agent's response was not in the expected format."
                        )

                except Exception as e:
                    logger.error(
                        f"Chat Response: Processing user query '{user_query}' failed. Error: {e}",
                        exc_info=True,
                    )
                    st.error(f"Processing your query '{user_query}' failed. Error: {e}")
                    