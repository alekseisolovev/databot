# DataBot

A LangGraph-based AI agent designed to automate data science workflows and analytical tasks, with Langfuse integration for performance evaluation and monitoring.

## Features

* The AI agent can answer questions about the provided dataset and return results directly as dataframes or series.
* It can create visual plots to help explore the data in more detail.

## Prerequisites

Before running the app, create an `.env` file in the project's root directory.

1.  **Google API Key:** Add your Google API key.
    ```bash
    GOOGLE_API_KEY="AI..."
    ```

2.  **Langfuse Tracing (Optional):** To enable observability and tracing with Langfuse, add your project keys.
    ```bash
    LANGFUSE_PUBLIC_KEY="pk-lf-..."
    LANGFUSE_SECRET_KEY="sk-lf-..."
    ```

## Running the App

To run the Streamlit app using Docker Compose, follow these steps.

1. In the project directory, build and start the service:
    ```bash
    docker-compose up --build app
    ```
2. Open your browser and go to: `http://localhost:8501/`

3. To stop the app, press `Ctrl+C` in the terminal.

## Running Tests

To run the automated tests for the agent, use the following command: 
```bash
docker-compose run --rm test
```
