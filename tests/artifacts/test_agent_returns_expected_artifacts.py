import pandas as pd

from src.agent import Agent


def initialize_agent(df: pd.DataFrame) -> Agent:
    """A helper function to initialize the agent for testing."""
    return Agent(df)


def test_agent_returns_expected_series():
    """
    Tests that the agent correctly processes a query that should return a pandas Series.
    """
    df = pd.read_csv("tests/data/iris.csv")
    agent = initialize_agent(df)
    target = df.groupby("Species")["PetalWidthCm"].mean()
    user_input = "What is the average PetalWidthCm for each species?"

    agent.invoke(user_input)
    last_message = agent.get_messages()[-1]
    output = last_message.additional_kwargs["dataframe_artifact"]

    pd.testing.assert_series_equal(target, output)


def test_agent_returns_expected_dataframe():
    """
    Tests that the agent correctly processes a query that should return a pandas DataFrame.
    """
    df = pd.read_csv("tests/data/iris.csv")
    agent = initialize_agent(df)
    target = df.describe()
    user_input = "What is the statistical summary of the dataset?"

    agent.invoke(user_input)
    last_message = agent.get_messages()[-1]
    output = last_message.additional_kwargs["dataframe_artifact"]

    pd.testing.assert_frame_equal(target, output)
