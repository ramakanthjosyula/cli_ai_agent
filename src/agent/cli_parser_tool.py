from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from src.config import LLM_MODEL_PATH
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from src.models.models_loader import load_llm


# Load the locally downloaded LLaMA-3.1 model
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_PATH, 
    torch_dtype=torch.float16,  # Use FP16 for efficiency
    device_map="auto"  # Auto-select best device (CPU/GPU)
)

# Create an inference pipeline
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1024,
    temperature=0.2
)

llm = load_llm()

# Define output structure
response_schemas = [
    ResponseSchema(name="parameters", description="Extracted or verified parameters in dictionary format")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Define prompt template
prompt = PromptTemplate(
    template="""
    Given the following CLI output:

    ```
    {cli_text}
    ```

    Verify the following parameters:

    ```
    {verification_params}
    ```

    If `find_out_flag` is True, extract values instead of verifying.

    Respond in structured format: {format_instructions}
    """,
    input_variables=["cli_text", "verification_params"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

# Define LLM Chain
cli_parser_chain = LLMChain(llm=llm, prompt=prompt)

def parse_cli_output(inputs):
    """
    Uses the LLM chain to extract or verify CLI output.
    """
    return cli_parser_chain.run(inputs)
