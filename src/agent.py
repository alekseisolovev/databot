import io

import pandas as pd
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


def get_dataframe_schema(df: pd.DataFrame) -> str:
    """Generates a string representation of the DataFrame's schema."""
    buffer = io.StringIO()
    df.info(buf=buffer, memory_usage=False, verbose=True, show_counts=True)
    return buffer.getvalue()
    