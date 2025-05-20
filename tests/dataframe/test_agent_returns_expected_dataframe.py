import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent import create_agent_graph, get_dataframe_schema, get_system_prompt


def test_agent_returns_expected_dataframe():

    df = pd.read_csv("tests/data/iris.csv")

    agent = create_agent_graph(df)
    dataframe_schema = get_dataframe_schema(df)
    system_prompt = get_system_prompt(dataframe_schema)

    initial_state = {"messages": [SystemMessage(content=system_prompt)]}
    user_input = "What is the average PetalWidthCm for each species?"
    messages = initial_state["messages"] + [HumanMessage(content=user_input)]

    target = df.groupby("Species")["PetalWidthCm"].mean()

    response = agent.invoke({"messages": messages})
    output = response["messages"][-1].additional_kwargs["dataframe"]

    assert target.equals(output)
    