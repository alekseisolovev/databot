import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent import create_agent_graph, get_dataframe_schema, get_system_prompt


def initialize_agent(df):

    agent = create_agent_graph(df)
    dataframe_schema = get_dataframe_schema(df)
    system_prompt = get_system_prompt(dataframe_schema)

    return agent, system_prompt


def test_agent_returns_expected_series():

    df = pd.read_csv("tests/data/iris.csv")
    agent, system_prompt = initialize_agent(df)

    target = df.groupby("Species")["PetalWidthCm"].mean()

    initial_state = {"messages": [SystemMessage(content=system_prompt)]}
    user_input = "What is the average PetalWidthCm for each species?"
    messages = initial_state["messages"] + [HumanMessage(content=user_input)]

    response = agent.invoke({"messages": messages})
    output = response["messages"][-1].additional_kwargs["dataframe_artifact"]

    pd.testing.assert_series_equal(target, output)


def test_agent_returns_expected_dataframe():

    df = pd.read_csv("tests/data/iris.csv")
    agent, system_prompt = initialize_agent(df)

    target = df.describe()

    initial_state = {"messages": [SystemMessage(content=system_prompt)]}
    user_input = "What is the statistical summary of the dataset?"
    messages = initial_state["messages"] + [HumanMessage(content=user_input)]

    response = agent.invoke({"messages": messages})
    output = response["messages"][-1].additional_kwargs["dataframe_artifact"]

    pd.testing.assert_frame_equal(target, output)
    