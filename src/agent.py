import io
import logging
from typing import Optional, Tuple, Union

import matplotlib.figure
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
    """Get DataFrame schema as a string."""
    buffer = io.StringIO()
    df.info(buf=buffer, memory_usage=False, verbose=True, show_counts=True)
    return buffer.getvalue()


def get_system_prompt(dataframe_schema: str) -> str:
    """Get system prompt for the agent."""
    return f"""
You are a helpful AI assistant for data analysis.
You have access to a pandas DataFrame, referred to as 'df', which you can interact with using Python code.

## General Behavior
- Based on the user's question, determine whether to respond directly or use the 'run_dataframe_query' tool.
- If you use the tool, the result (a pandas DataFrame/Series, a Matplotlib figure, or other object) will be returned to you as an observation.
- If the tool returns a DataFrame/Series or a plot/figure, assume the data/plot will be shown separately and **do not repeat or reprint it.**
    - Instead, write a general summary or overview, such as: “Here are the results.”, or "Here is the requested plot."
    - You may optionally include commentary like: “Let me know if you’d like help interpreting the results.”
- If the tool returns a string, list, number, or boolean, respond using the result explicitly.
- Your answer should be clear, concise, and focused on **helping the user understand the data**.

## Tool Use
- If a query is needed, write a valid pandas expression using standard syntax.
- For plotting, ensure your query returns a `matplotlib.figure.Figure` object.
- Examples of valid queries:
    - View the first 5 rows: "df.head()"
    - Filter rows where 'age' is greater than 30: "df[df['age'] > 30]"
    - Count unique values in the 'gender' column: "df['gender'].value_counts()"
    - Get summary statistics for all numeric columns: "df.describe()"
    - Find rows with missing values in 'income': "df[df['income'].isnull()]"

After executing any tool-based query, interpret the results and give a clear, user-friendly answer.

## End-to-end Examples

1.  **User**: What are the column names?
    **Tool**: `list(df.columns)` → `['age', 'income', 'gender']`
    **Response**: The DataFrame contains the columns: age, income, and gender.

2.  **User**: Show the dataset's statistical summary.
    **Tool**: `df.describe()` → `<DataFrame>`
    **Response**: Here are the summary statistics for the numeric columns.

3.  **User**: Which values are most common in the gender column?
    **Tool**: `df['gender'].value_counts()` → `<Series>`
    **Response**: These are the most frequent values in the gender column.

4.  **User**: Are there any missing values in the income column?
    **Tool**: `df['income'].isnull().sum()` → `12`
    **Response**: Yes, there are 12 missing values in the income column.

5.  **User**: Show all rows where age is greater than 50.
    **Tool**: `df[df['age'] > 50]` → `<DataFrame>`
    **Response**: Here are the rows for people older than 50.

6.  **User**: Plot a histogram of the 'age' column.
    **Tool**: `df['age'].plot(kind='hist').figure` → `<Figure>`
    **Response**: Here is a histogram of the 'age' column.

7.  **User**: Can you show me a bar chart of the 'gender' counts?
    **Tool**: `df['gender'].value_counts().plot(kind='bar').figure` → `<Figure>`
    **Response**: Here's a bar chart showing the distribution of genders.

8.  **User**: Create a scatter plot of 'income' vs 'age'.
    **Tool**: `df.plot.scatter(x='income', y='age').figure` → `<Figure>`
    **Response**: Here is a scatter plot of income versus age.

9.  **User**: Generate a box plot for 'salary'.
    **Tool**: `df['salary'].plot(kind='box').figure` → `<Figure>`
    **Response**: Here is the box plot for salary.

10. **User**: Plot the distribution of 'score' using a density plot.
    **Tool**: `df['score'].plot(kind='density').figure` → `<Figure>`
    **Response**: Here's a density plot for the 'score' column.

-----------------
DataFrame Schema:
{dataframe_schema}
-----------------
"""


class Agent:

    def __init__(self, df: pd.DataFrame):

        self.df = df
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20")
        self.graph = self._build_graph()

    def _build_graph(self):

        @tool(response_format="content_and_artifact")
        def run_dataframe_query(
            query: str,
        ) -> Tuple[
            str, Optional[Union[pd.DataFrame, pd.Series, matplotlib.figure.Figure]]
        ]:
            """
            Executes a Python query or operation on a pandas DataFrame.

            Returns:
                A tuple containing:
                - The primary textual output (str): This can be a string representation of a scalar, a list or an error message.
                - An optional artifact (pd.DataFrame, pd.Series, or matplotlib.figure.Figure)
                if the query result is a complex object; otherwise, None.
            """
            try:
                result = eval(query, {"df": self.df, "pd": pd}, {})
                artifact = None

                if isinstance(
                    result, (pd.DataFrame, pd.Series, matplotlib.figure.Figure)
                ):
                    content = f"Query '{query}' returned {type(result)}."
                    artifact = result
                else:
                    content = f"Query '{query}' returned {str(result)}."
                logger.info(f"Tool 'run_dataframe_query': {content}")

                return content, artifact

            except Exception as e:
                error_message = f"Error executing query '{query}': {str(e)}"
                logger.error(
                    f"Tool Node 'run_dataframe_query': {error_message}", exc_info=True
                )
                return error_message, None

        tools = [run_dataframe_query]
        tool_node = ToolNode(tools)
        self.model = self.model.bind_tools(tools)

        def agent_node(state: MessagesState):
            response = self.model.invoke(state["messages"])

            if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
                last_tool_message = state["messages"][-1]
                if last_tool_message.artifact is not None:
                    artifact = last_tool_message.artifact
                    response.additional_kwargs = response.additional_kwargs or {}
                    key = None
                    if isinstance(artifact, (pd.DataFrame, pd.Series)):
                        key = "dataframe_artifact"
                    elif isinstance(artifact, matplotlib.figure.Figure):
                        key = "figure_artifact"
                    if key:
                        response.additional_kwargs[key] = artifact
                        logger.info(
                            f"Agent Node: Attached {type(artifact)} artifact to AIMessage."
                        )
                    else:
                        logger.warning(
                            f"Agent Node: Unrecognized artifact type: {type(artifact)}"
                        )
                else:
                    logger.info("Agent Node: ToolMessage returned no artifact.")

            return {"messages": [response]}

        def should_continue(state: MessagesState):
            last_message = state["messages"][-1]
            if not isinstance(last_message, AIMessage):
                logger.error(
                    f"Router Node: Expected AIMessage, got {type(last_message)}. Content: '{last_message.content}'. Ending graph."
                )
                return END

            if last_message.tool_calls:
                tool_name = last_message.tool_calls[0]["name"]
                tool_args = last_message.tool_calls[0]["args"]
                logger.info(
                    f"Router Node: AI requests tool '{tool_name}' with args: {tool_args}."
                )
                return "tools"
            else:
                logger.info(
                    f"Router Node: AI provided final response. Content: '{last_message.content}'. Ending graph."
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
            logger.info("Graph compiled.")
            return graph
        except Exception as e:
            logger.error(f"Graph compilation failed. Error: {e}", exc_info=True)
            raise

    def invoke(self, messages: list) -> Optional[AIMessage]:
        return self.graph.invoke(messages)
        