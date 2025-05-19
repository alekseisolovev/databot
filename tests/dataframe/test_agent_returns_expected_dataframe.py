import pandas as pd

from src.agent import create_agent_graph, get_dataframe_schema, get_system_prompt


def test_agent_returns_expected_dataframe():
    df = pd.read_csv("tests/data/iris.csv")
    assert len(df)
    