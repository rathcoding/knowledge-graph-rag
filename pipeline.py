import os
import logging
import dotenv
dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)
logging.info('Starting the data pipeline...')

# Get list of PDF files from 'files' folder
logging.info('Getting list of PDF files from "files" folder')
files_path = 'files'
files = [files_path+'/'+file for file in os.listdir(files_path) if file.endswith('.pdf')]
logging.info(f'List of PDF files: {files}')

# Instantiate the token text splitter
logging.info('Instantiating the token text splitter')
from langchain.text_splitter import TokenTextSplitter
splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)

# Split the PDFs into chunks as Documents
logging.info('Splitting the PDFs into chunks as Documents')
from langchain_community.document_loaders import PyPDFLoader
documents = []

for file in files:
    # Load the PDF file
    pdf_loader = PyPDFLoader(file_path=file, extract_images=False)
    # Split the PDF into Documents
    files_documents = pdf_loader.load_and_split(text_splitter=splitter)
    # Add the Documents to the list
    documents.extend(files_documents)
    logging.info(f'Loaded and split {file} into {len(files_documents)} Documents')

# Instantiate LLM to use with the LLMGraphTransformer
logging.info('Instantiating LLM to use with the LLMGraphTransformer')
from langchain_community.llms import Ollama
llm=Ollama(model='llama3', temperature=0.0)

# To use a local LLM we need to create a chat_prompt to solve the AttributeError from the LLMGraphTransformer

# Create a system message to provide the LLM with the instructions
logging.info('Creating a chat_prompt to provide the LLM with the instructions and examples')
from langchain_experimental.graph_transformers.llm import SystemMessage
system_prompt = """
You are a data scientist working for the police and you are building a knowledge graph database. 
Your task is to extract information from data and convert it into a knowledge graph database.
Provide a set of Nodes in the form [head, head_type, relation, tail, tail_type].
It is important that the head and tail exists as nodes that are related by the relation.
If you can't pair a relationship with a pair of nodes don't add it.
When you find a node or relationship you want to add try to create a generic TYPE for it that describes the entity you can also think of it as a label.
You must generate the output in a JSON format containing a list with JSON objects. Each object should have the keys: "head", "head_type", "relation", "tail", and "tail_type".
"""

system_message = SystemMessage(content=system_prompt)

# Create a human message to combine with the SystemMessage
# We need:
# 1) A parser to provide the LLM with the format instructions
# 2) Examples to provide the LLM with the context we want to extract
# 3) A list of nodes and relationships to provide the LLM with the context we want to extract

# 1) The parser
# To instantiate a parser we need a Pydantic class.
# Instead of using the default langchain_experimental.graph_transformers.llm.UnstructuredRelation,
# we'll build our own class to provide more crime-related context instructions
from langchain_core.pydantic_v1 import BaseModel, Field

class UnstructuredRelation(BaseModel):
    head: str = Field(
        description=(
            "extracted head entity like Person, Crime, Object, Vehicle, Location, etc."
            "Must use human-readable unique identifier."
        )
    )
    head_type: str = Field(
        description="type of the extracted head entity like Person, Crime, Object, Vehicle, etc"
    )
    relation: str = Field(description="relation between the head and the tail entities")
    tail: str = Field(
        description=(
            "extracted head entity like Person, Crime, Object, Vehicle, Location, etc."
            "Must use human-readable unique identifier."
        )
    )
    tail_type: str = Field(
        description="type of the extracted head entity like Person, Crime, Object, Vehicle, etc"
    )

# Instantiate the parser with our Pydantic class to provide the LLM with the format instructions
from langchain_experimental.graph_transformers.llm import JsonOutputParser
parser = JsonOutputParser(pydantic_object=UnstructuredRelation)

# 2) The examples
examples = [
    {
        "text": (
            "Michael Johnson was mugged at knife-point by two assailants on 5th Avenue. "
            "They took his wallet and phone."
        ),
        "head": "Michael Johnson",
        "head_type": "Person",
        "relation": "VICTIM_OF",
        "tail": "Mugging",
        "tail_type": "Crime",
    },
    {
        "text": (
            "Michael Johnson was mugged at knife-point by two assailants on 5th Avenue. "
            "They took his wallet and phone."
        ),
        "head": "5th Avenue",
        "head_type": "Location",
        "relation": "SCENE_OF",
        "tail": "Mugging",
        "tail_type": "Crime",
    },
    {
        "text": (
            "Sarah Connor witnessed a mugging on 5th Avenue where Michael Johnson was attacked. "
            "She saw the assailants flee in a black car."
        ),
        "head": "Sarah Connor",
        "head_type": "Person",
        "relation": "WITNESS_OF",
        "tail": "Mugging",
        "tail_type": "Crime",
    },
    {
        "text": (
            "John Doe was caught selling illegal drugs in Central Park. "
            "He was arrested by undercover officers."
        ),
        "head": "John Doe",
        "head_type": "Person",
        "relation": "SUSPECT_IN",
        "tail": "Drug Trafficking",
        "tail_type": "Crime",
    },
    {
        "text": (
            "John Doe was caught selling illegal drugs in Central Park. "
            "He was arrested by undercover officers."
        ),
        "head": "Central Park",
        "head_type": "Location",
        "relation": "SCENE_OF",
        "tail": "Drug Trafficking",
        "tail_type": "Crime",
    },
    {
        "text": (
            "Emily Clark was assaulted in a parking lot near her office on Elm Street. "
            "The assailant attempted to steal her car but fled when she screamed."
        ),
        "head": "Emily Clark",
        "head_type": "Person",
        "relation": "VICTIM_OF",
        "tail": "Assault",
        "tail_type": "Crime",
    },
    {
        "text": (
            "Emily Clark was assaulted in a parking lot near her office on Elm Street. "
            "The assailant attempted to steal her car but fled when she screamed."
        ),
        "head": "Elm Street",
        "head_type": "Location",
        "relation": "SCENE_OF",
        "tail": "Assault",
        "tail_type": "Crime",
    },
    {
        "text": (
            "James Smith was identified as the suspect in the assault on Emily Clark. "
            "He was later arrested by the police."
        ),
        "head": "James Smith",
        "head_type": "Person",
        "relation": "SUSPECT_IN",
        "tail": "Assault",
        "tail_type": "Crime",
    },
    {
        "text": (
            "Laura Adams witnessed the assault on Emily Clark and provided a description of the assailant to the police."
        ),
        "head": "Laura Adams",
        "head_type": "Person",
        "relation": "WITNESS_OF",
        "tail": "Assault",
        "tail_type": "Crime",
    },
    {
        "text": (
            "David Brown attempted to murder Lisa White by poisoning her drink at a party on Pine Street. "
            "She was hospitalized but survived the attack."
        ),
        "head": "David Brown",
        "head_type": "Person",
        "relation": "SUSPECT_IN",
        "tail": "Attempted Murder",
        "tail_type": "Crime",
    },
    {
        "text": (
            "David Brown attempted to murder Lisa White by poisoning her drink at a party on Pine Street. "
            "She was hospitalized but survived the attack."
        ),
        "head": "Lisa White",
        "head_type": "Person",
        "relation": "VICTIM_OF",
        "tail": "Attempted Murder",
        "tail_type": "Crime",
    },
    {
        "text": (
            "David Brown attempted to murder Lisa White by poisoning her drink at a party on Pine Street. "
            "She was hospitalized but survived the attack."
        ),
        "head": "Pine Street",
        "head_type": "Location",
        "relation": "SCENE_OF",
        "tail": "Attempted Murder",
        "tail_type": "Crime",
    },
    {
        "text": (
            "Mark Thompson witnessed David Brown putting something in Lisa White's drink at the party. "
            "He reported this to the police."
        ),
        "head": "Mark Thompson",
        "head_type": "Person",
        "relation": "WITNESS_OF",
        "tail": "Attempted Murder",
        "tail_type": "Crime",
    }
]

# 3) The list of nodes and relationships
# to be coded and experimented with...

# Instantiate the human prompt template using the examples and Pydantic class with format instructions
from langchain_experimental.graph_transformers.llm import PromptTemplate
human_prompt = PromptTemplate(
    template="""
Examples:
{examples}

For the following text, extract entities and relations as in the provided example.
{format_instructions}\nText: {input}""",
    input_variables=["input"],
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
        "node_labels": None,
        "rel_types": None,
        "examples": examples,
    },
)

# Instantiate the human message prompt to provide the LLM with the instructions and examples
from langchain_experimental.graph_transformers.llm import HumanMessagePromptTemplate
human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

# Create a chat_prompt combining the system mand human messages to use with the LLMGraphTransformer
from langchain_experimental.graph_transformers.llm import ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message, human_message_prompt]
)

# Instantiate the LLMGraphTransformer that will extract the entities and relationships from the Documents
logging.info('Instantiating the LLMGraphTransformer that will extract the entities and relationships from the Documents')
from langchain_experimental.graph_transformers import LLMGraphTransformer
llm_transformer = LLMGraphTransformer(llm=llm, prompt=chat_prompt)

# Convert the Documents into Graph Documents
# This is the heavy computation part...
logging.info('Converting the Documents into Graph Documents...')
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Instantiate the Neo4JGraph to persist the data
logging.info('Instantiating the Neo4JGraph to persist the data')
from langchain_community.graphs import Neo4jGraph
graph = Neo4jGraph()

# Persist the Graph Documents into the Neo4JGraph
logging.info('Persisting the Graph Documents into the Neo4JGraph')
graph.add_graph_documents(
  graph_documents,
  baseEntityLabel=True,
  include_source=True
)

logging.info('Data pipeline completed successfully!')
