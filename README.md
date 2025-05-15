# DataBot

A LangGraph-based AI agent designed to assist with data science workflows and analytical tasks.

## Prerequisites

Before running the app, ensure you create an `.env` file in the project root and add your Google API key:
```bash
GOOGLE_API_KEY=your_api_key_here
```
## Running the App

To run the Streamlit app using Docker Compose, follow these steps.

1. In the project directory, build and start the service using:
    ```bash
    docker compose up --build
    ```
2. Go to: http://localhost:8501/

3. To stop the app, press `Ctrl+C` in the terminal.
