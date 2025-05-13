import io
import logging

import pandas as pd
from langchain_core.messages import AIMessage, SystemMessage
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


def create_agent_graph(df: pd.DataFrame):
    """Creates and compiles the LangGraph agent."""

    @tool
    def run_dataframe_query(query: str) -> str:
        """
        Executes a Python query or operation on the pandas DataFrame.
        Use this to get specific data, summaries, or perform calculations.
        The query should be a string that can be evaluated.
        """
        try:
            result = eval(query, {"df": df, "pd": pd}, {})
            logger.info(
                f"Tool Message: Query '{query}' executed. Result: {str(result)}"
            )
            return str(result)
        except Exception as e:
            logger.error(
                f"Tool Message: Error executing query: '{query}'. Error: {str(e)}"
            )
            return f"Error executing query: '{query}'. Error: {str(e)}"

    tools = [run_dataframe_query]
    tool_node = ToolNode(tools)

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash").bind_tools(tools)

    def agent_node(state: MessagesState):
        logger.debug(f"AI Message: Processing messages: {state['messages']}")
        response = model.invoke(state["messages"])
        logger.debug(f"AI Message: Received response from model: {response}")
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            logger.error(
                f"Expected AIMessage but got: '{last_message.content}'. Ending graph run."
            )
            return END
        if last_message.tool_calls:
            tool_name = last_message.tool_calls[0]["name"]
            tool_args = last_message.tool_calls[0]["args"]
            logger.info(
                f"AI Message: Calling tool '{tool_name}' with args: {tool_args}"
            )
            return "tools"
        else:
            logger.info(f"AI Message: Sending response: {last_message.content}")
            return END

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    builder.add_edge("tools", "agent")

    graph = builder.compile()
    logger.info("Agent graph compiled successfully.")
    return graph
    