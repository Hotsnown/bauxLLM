import pandas as pd
import json
import argparse

from .config.config import DATA_STORE_DIR
from .data.knowledge_base import load_vector_store

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

vector_store = load_vector_store(DATA_STORE_DIR)

with open("./src/config/questionnaire.json", "r") as questionnaire_file:
    questionnaire_content = questionnaire_file.read()
    questionnaire = json.loads(questionnaire_content)

system_template="""You are a lawyer who specialised in lease agreements."""

user_template="""----------------
Answer the question based on the context below. Keep the answer short. Respond "Unsure about answer" if not sure about the answer.

Context: {summaries}

Question: {question}

Answer:
"""

def generate_response(system_template, user_template):
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template)
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {
        "prompt": prompt
        }
    
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=256)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        reduce_k_below_max_tokens=True    
        )
    
    return chain

def extract(query):
    chain = generate_response(system_template, user_template)
    result = chain(query)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate responses based on the provided query or a default questionnaire.')
    parser.add_argument('--query', type=str, help='A query string to generate response for.')
    args = parser.parse_args()

    results = []

    if args.query:  # If the 'query' argument is provided.
        chain = generate_response(system_template, user_template)
        result = chain(args.query)
        answer = result["answer"]
        results.append([args.query, answer])
        print(answer)

    else:  # If the 'query' argument is not provided, load questions from the questionnaire.
        for section in questionnaire["sections"]:
            for question in section["questions"]:
                query = question["question"]
                chain = generate_response(system_template, user_template)
                result = chain(query)
                answer = result["answer"]
                sources = result.get("source_documents", [])[:4]  
                results.append([query, answer]) 

        df = pd.DataFrame(results, columns=["query", "result"])
        df.to_excel("./data/results.xlsx", index=False)
