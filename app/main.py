import pandas as pd
import streamlit as st

st.markdown(
    """
    <h1 style='text-align: center;'>DataBot</h1>
    <h4 style='text-align: center; color: gray;'>AI Agent for Data Science</h4>
""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Upload your file here:", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
        