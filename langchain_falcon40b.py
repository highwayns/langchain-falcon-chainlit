from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import PromptTemplate, LLMChain, HuggingFacePipeline
import os

from dotenv import load_dotenv
import chainlit as cl

# Load environment variables from .env file
load_dotenv()


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

model_id = "tiiuae/falcon-40b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
#model = AutoModelForCausalLM.from_pretrained(
#    model_id,
#    torch_dtype=torch.bfloat16,
#    trust_remote_code=True,
#    device_map={"": 0},
#)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # --- Choosing between 4, 8, and 16 bit --- #
    # 8bit: ~50GB GPU memory, fastest
    # 4bit: ~25GB GPU memory, slowest 
    # 16bit: ~100GB GPU memory, slow
    load_in_8bit=True, # torch_dtype=torch.bfloat16 or load_in_4bit=True
    trust_remote_code=True,
    device_map="auto",
)


pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1200,
    temperature=0.6,
)

llm = HuggingFacePipeline(pipeline=pipeline)

template = """
You are a helpful AI assistant and provide the answer for the question asked politely.

{question}
"""

@cl.langchain_factory(use_async=False)
def factory():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain

