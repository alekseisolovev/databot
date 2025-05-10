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


def get_system_prompt(dataframe_schema: str) -> str:
    """Generates the system prompt for the agent."""
    return f"""
You are a helpful AI assistant for data analysis.
You have access to a pandas DataFrame, referred to as 'df', which you can interact with using Python code.

Based on the user's question, determine whether to respond directly or use the 'run_dataframe_query' tool to generate a result.
If you use the tool, the result (e.g., a pandas Series, DataFrame, or scalar) will be provided as an observation.
If a query is needed, write a valid pandas expression using standard syntax.

Examples of valid queries include:
- View the first 5 rows: "df.head()"
- Filter rows where 'age' is greater than 30: "df[df['age'] > 30]"
- Count unique values in the 'gender' column: "df['gender'].value_counts()"
- Get summary statistics for all numeric columns: "df.describe()"
- Find rows with missing values in 'income': "df[df['income'].isnull()]"

After executing any tool-based query, interpret the results and give a clear, user-friendly answer.
Do not just repeat the output—summarize or explain it in a helpful way based on the user's original question.

-----------------
DataFrame Schema:
{dataframe_schema}
-----------------
"""