from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from src.config import LLM_MODEL_PATH

def load_llm():
    # Load your model (replace with your local model path if necessary)
    text_gen_pipeline = pipeline("text-generation", model=LLM_MODEL_PATH)
    return text_gen_pipeline
