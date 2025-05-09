import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

st.set_page_config(layout="wide")
st.markdown(
    """
    <h1 style='text-align: center;'>DataBot</h1>
    <h4 style='text-align: center; color: gray;'>AI Agent for Data Science</h4>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None

with st.sidebar:
    st.header("Upload your CSV file")
    uploaded_file = st.file_uploader("", type=["csv"])

    def reset_session_state(df):
        st.session_state.dataframe = df

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if st.session_state.get(
                "dataframe"
            ) is None or not st.session_state.dataframe.equals(df):
                reset_session_state(df)

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            reset_session_state(None)

        if st.session_state.dataframe is not None:
            st.header("Data Preview")
            st.dataframe(st.session_state.dataframe.head(), height=200)
            st.success("CSV successfully uploaded. You can now chat with the agent.")

for message in st.session_state.messages:
    if message.type == "human":
        st.chat_message("user").write(message.content)
    elif message.type == "agent":
        st.chat_message("assistant").write(message.content)

if user_query := st.chat_input("Ask something about your data."):
    if st.session_state.dataframe is None:
        st.warning("Please upload a CSV file first.")
    else:
        st.session_state.messages.append(HumanMessage(content=user_query))
        st.chat_message("user").write(user_query)
        