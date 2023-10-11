import pandas as pd
import json
import argparse
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from .config.config import DATA_STORE_DIR
from .data.knowledge_base import load_vector_store

from enum import Enum

class Answer(Enum):
    YES = "/yes/"
    NO = "/no/"

vector_store = load_vector_store(DATA_STORE_DIR)
file_path = "./data/results.xlsx"

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

    chain = LLMChain(llm=llm, prompt=prompt)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"score_threshold": .5}),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True,
        reduce_k_below_max_tokens=True    
        )
    
    return chain

def self_consistency_prompting(query, iterations=5):
    # Consolidate votes and answers in a dictionary
    votes = {Answer.YES: 0, Answer.NO: 0}
    answers = {Answer.YES: [], Answer.NO: []}

    for _ in range(iterations):
        chain = generate_response(system_template, user_template)
        result = chain(query)
        answer = result["answer"]

        # Count votes and record answers
        for vote_key in votes.keys():
            if vote_key.value in answer:
                votes[vote_key] += 1
                answers[vote_key].append(answer)

    # Return the majority answer
    for vote_key, count in votes.items():
        if count > iterations / 2:
            return result, answers[vote_key][0]

    return "No majority answer found."

def extract(query):
    chain = generate_response(system_template, user_template)
    result = chain(query)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate responses based on the provided query or a default questionnaire.')
    parser.add_argument('--query', type=str, help='A query string to generate response for.')
    args = parser.parse_args()

    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} has been deleted.")
    else:
        print(f"The file {file_path} does not exist.")

    results = []

    if args.query:  # If the 'query' argument is provided.
        answer = self_consistency_prompting(args.query)
        print(answer)

    else:  # If the 'query' argument is not provided, load questions from the questionnaire.
        for section in questionnaire["sections"]:
            for question in section["questions"]:
                result, majority_answer = self_consistency_prompting(question)
                sources = [doc.dict()["page_content"] for doc in result['source_documents']]
                results.append([question, majority_answer, sources]) 

        df = pd.DataFrame(results, columns=["question", "answer", "contexts"])
        df.to_excel(file_path, index=False)
