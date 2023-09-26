import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import uuid

from ..config.config import DATA_STORE_DIR

def load_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def generate_documents_from_texts(texts):
    return [Document(
        page_content=text,
        metadata={"source": str(uuid.uuid4())}
        ) for text in texts]


def save_embeddings_to_vector_store(documents, directory):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(directory)


def load_vector_store(directory):
    embeddings = OpenAIEmbeddings()
    if os.path.exists(directory):
        return FAISS.load_local(directory, embeddings)
    else:
        print(f"Missing files. Upload index.faiss and index.pkl files to {directory} directory first")
        return None


def search_and_display_results(vector_store, query):
    search_result = vector_store.similarity_search_with_score(query)

    line_separator = "\n"
    print(f"""
    Search results:{line_separator}
    {line_separator.join([
    f'''
    Score:{line_separator}{r[1]}{line_separator}
    Content:{line_separator}{r[0].page_content}{line_separator}
    '''
    for r in search_result
    ])}
    """)


def main():
    files = [load_text_from_file(f"./data/sections/{file}") for file in os.listdir("./data/sections")]
    documents = generate_documents_from_texts(files)
    save_embeddings_to_vector_store(documents, DATA_STORE_DIR)
    vector_store = load_vector_store(DATA_STORE_DIR)

    if vector_store:
        search_and_display_results(vector_store, "Who is the lessee ?")


if __name__ == '__main__':
    main()