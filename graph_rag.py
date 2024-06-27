import os
import logging
import dotenv
dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)
logging.info('Starting up the Knowledge Graph RAG...')

# Instantiate the Neo4J connector
logging.info(f'Instantiating the Neo4J connector for: { os.getenv("NEO4J_URI") }')
from langchain_community.graphs import Neo4jGraph
graph = Neo4jGraph()

# Instantiate LLM to use with the Graph RAG
logging.info('Instantiating LLM to use with the LLMGraphTransformer')
from langchain_community.llms import Ollama
llm=Ollama(model='llama3', temperature=0.0)

# Instantiate the langchain Graph RAG with the Neo4J connector and the LLM
from langchain.chains import GraphCypherQAChain
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)

logging.info('Knowledge Graph RAG is ready to go!')
logging.info('='*50)

def main():
    logging.info('Type "exit" to quit the program.')
    while True:
        question = input('\nAsk me a question: ')
        if question == 'exit':
            break
        result = chain.invoke({"query": question})
        if result['result']:
            print(result['result'])
        else:
            print(result)


if __name__ == '__main__':
    main()
