from langchain.agents import initialize_agent, Tool
from src.agent.cli_parser_tool import parse_cli_output
from src.models_loader import load_llm

llm = load_llm()

# Define the CLI parser tool
cli_parser_tool = Tool(
    name="CLIParser",
    func=parse_cli_output,
    description="Parses CLI output and verifies/extracts parameters."
)

# Create the agent with this tool
agent = initialize_agent(
    tools=[cli_parser_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)


def process_cli_with_agent(cli_text, verification_params, find_out_flag):
    """
    Calls the LangChain agent to process CLI output.
    """
    return agent.run({"cli_text": cli_text, "verification_params": verification_params, "find_out_flag": find_out_flag})
