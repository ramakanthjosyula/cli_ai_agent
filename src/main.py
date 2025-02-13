from fastapi import FastAPI
from pydantic import BaseModel
from src.agent.langchain_agent import process_cli_with_agent

app = FastAPI(title="CLI AI Agent", description="An AI-powered CLI output interpreter", version="1.0")


class CLIRequest(BaseModel):
    cli_text: str
    verification_params: dict
    find_out_flag: bool = False


@app.post("/process_cli/")
async def process_cli(request: CLIRequest):
    """
    API endpoint to process CLI output.
    - `cli_text`: Raw CLI output.
    - `verification_params`: Dictionary of expected values.
    - `find_out_flag`: If True, extracts values instead of verifying.
    """
    result = process_cli_with_agent(request.cli_text, request.verification_params, request.find_out_flag)
    return {"result": result}
