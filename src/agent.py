import io
import logging
from typing import Optional, Tuple, Union

import pandas as pd
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

## General Behavior
- Based on the user's question, determine whether to respond directly or use the 'run_dataframe_query' tool.
- If you use the tool, the result (a pandas DataFrame/Series or other object) will be returned to you as an observation.
- If the tool returns a DataFrame/Series, assume the data will be shown separately and **do not repeat or reprint it.**
    - Instead, write a general summary or overview, such as: “Here are the results.”, etc.   
    - You may optionally include commentary like: “Let me know if you’d like help interpreting the results.”
- If the tool returns a string, list, number, or boolean, respond using the result explicitly.
- Your answer should be clear, concise, and focused on **helping the user understand the data**.

## Tool Use
- If a query is needed, write a valid pandas expression using standard syntax.
- Examples of valid queries:
    - View the first 5 rows: "df.head()"
    - Filter rows where 'age' is greater than 30: "df[df['age'] > 30]"
    - Count unique values in the 'gender' column: "df['gender'].value_counts()"
    - Get summary statistics for all numeric columns: "df.describe()"
    - Find rows with missing values in 'income': "df[df['income'].isnull()]"

After executing any tool-based query, interpret the results and give a clear, user-friendly answer.

## End-to-end Examples

1. **User**: What are the column names?  
   **Tool**: `list(df.columns)` → `['age', 'income', 'gender']`  
   **Response**: The DataFrame contains the columns: age, income, and gender.

2. **User**: Show the dataset's statistical summary.  
   **Tool**: `df.describe()` → `<DataFrame>`  
   **Response**: Here are the summary statistics for the numeric columns.

3. **User**: Which values are most common in the gender column?  
   **Tool**: `df['gender'].value_counts()` → `<Series>`  
   **Response**: These are the most frequent values in the gender column.

4. **User**: Are there any missing values in the income column?  
   **Tool**: `df['income'].isnull().sum()` → `12`  
   **Response**: Yes, there are 12 missing values in the income column.

5. **User**: Show all rows where age is greater than 50.  
   **Tool**: `df[df['age'] > 50]` → `<DataFrame>`  
   **Response**: Here are the rows for people older than 60.

-----------------
DataFrame Schema:
{dataframe_schema}
-----------------
"""


def create_agent_graph(df: pd.DataFrame):
    """Creates and compiles the LangGraph agent."""

    @tool(response_format="content_and_artifact")
    def run_dataframe_query(
        query: str,
    ) -> Tuple[str, Optional[Union[pd.DataFrame, pd.Series]]]:
        """
        Executes a Python query or operation on the pandas DataFrame.
        Returns a text summary and optionally a DataFrame/Series artifact if the query produces one.
        """
        try:
            result = eval(query, {"df": df, "pd": pd}, {})

            if isinstance(result, (pd.DataFrame, pd.Series)):
                content = f"Query executed: '{query}'. Result is a DataFrame/Series."
                logger.info(
                    f"Tool 'run_dataframe_query': Query '{query}' executed successfully. Result type: {type(result)}, Shape: {result.shape}"
                )
                return content, result
            else:
                content = f"Query executed: '{query}'. Result: {str(result)}."
                logger.info(
                    f"Tool 'run_dataframe_query': Query '{query}' executed successfully. Result: {str(result)}"
                )
                return content, None
        except Exception as e:
            error_message = f"Error executing query '{query}': {str(e)}"
            logger.error(f"Tool 'run_dataframe_query': {error_message}", exc_info=True)
            return error_message, None

    tools = [run_dataframe_query]
    tool_node = ToolNode(tools)

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash").bind_tools(tools)

    def agent_node(state: MessagesState):
        logger.debug(f"Agent Node: Processing messages: {state['messages']}")
        response_message = model.invoke(state["messages"])

        if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
            last_tool_message = state["messages"][-1]
            if last_tool_message.artifact is not None:
                artifact = last_tool_message.artifact
                if isinstance(artifact, (pd.DataFrame, pd.Series)):
                    if response_message.additional_kwargs is None:
                        response_message.additional_kwargs = {}
                    response_message.additional_kwargs["dataframe"] = artifact
                    logger.info(
                        f"Agent Node: Attached artifact to AIMessage. Type: {type(artifact)}, Shape: {getattr(artifact, 'shape', 'N/A')}"
                    )
                else:
                    logger.warning(
                        f"Agent Node: ToolMessage artifact was not None, but not a DataFrame/Series. Type: {type(artifact)}"
                    )
            else:
                logger.info("Agent Node: ToolMessage reported no artifact.")

        logger.debug(f"Agent Node: Model response received: {response_message.content}")
        return {"messages": [response_message]}

    def should_continue(state: MessagesState):
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            logger.error(
                f"Decision Node: Expected AIMessage, got {type(last_message)}. Content: '{last_message.content}'. Ending graph."
            )
            return END

        if last_message.tool_calls:
            tool_name = last_message.tool_calls[0]["name"]
            tool_args = last_message.tool_calls[0]["args"]
            logger.info(
                f"Decision Node: AI requests tool '{tool_name}' with args: {tool_args}. Routing to tools."
            )
            return "tools"
        else:
            logger.info(
                f"Decision Node: AI provided final response. Content: '{last_message.content}'. Ending graph."
            )
            return END

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "agent")

    try:
        graph = builder.compile()
        logger.info("Agent: Graph compiled successfully.")
        return graph
    except Exception as e:
        logger.error(f"Agent: Graph compilation failed. Error: {e}", exc_info=True)
        raise
        