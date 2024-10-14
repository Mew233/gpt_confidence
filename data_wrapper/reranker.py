# import re
# from langchain_community.document_loaders import TextLoader
# from langchain_community.embeddings import CohereEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
#
# COHERE_API_KEY = "IrrrvQoLE6MIiV30z8g0cKdWKSDIVgyUmCplbjCx"
# # Load the document
# documents = TextLoader("../postprocessed_gbm/PM336.txt").load()
#
# # Extracting raw text from document
# raw_text = documents[0].page_content
#
# # Split the raw text into smaller chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
# text_chunks = text_splitter.split_text(raw_text)
#
# # Create embeddings and FAISS retriever
# retriever = FAISS.from_texts(
#     text_chunks, CohereEmbeddings(model="embed-english-v3.0",cohere_api_key=COHERE_API_KEY,user_agent="my-app")
# ).as_retriever(search_kwargs={"k": 5})
#
# # Query to retrieve radiology-specific information
# query = "What is the patient's medication"
# docs = retriever.invoke(query)
#
# # Pretty print the most relevant radiology information
# def pretty_print_docs(docs):
#     for idx, doc in enumerate(docs):
#         print(f"\nDocument {idx + 1}:\n{doc.page_content}\n")
#
# pretty_print_docs(docs)
#
# # Optional: Further process the results to eliminate redundant information
# def remove_redundant_information(text):
#     # Removing identifiable information like names, MRNs, DOBs, etc.
#     text = re.sub(r"PATIENT NAME: .*?\n", "", text)
#     text = re.sub(r"MRN: \d+", "", text)
#     text = re.sub(r"DOB: \d{2}/\d{2}/\d{4}", "", text)
#     text = re.sub(r"Printed by .*? at .*?\n", "", text)
#     return text
#
# # Clean up and print the final refined information
# for idx, doc in enumerate(docs):
#     cleaned_text = remove_redundant_information(doc.page_content)
#     print(f"\nCleaned Document {idx + 1}:\n{cleaned_text}\n")

from langchain.retrievers import ContextualCompressionRetriever, CohereRagRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.chat_models import ChatCohere
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

user_query =  "When was Cohere started?"
API_KEY = "IrrrvQoLE6MIiV30z8g0cKdWKSDIVgyUmCplbjCx"
# Create cohere's chat model and embeddings objects
cohere_chat_model = ChatCohere(cohere_api_key="{API_KEY}")
cohere_embeddings = CohereEmbeddings(cohere_api_key="{API_KEY}",user_agent="my-app")
# Load text files and split into chunks, you can also use data gathered elsewhere in your application
raw_documents = TextLoader("../postprocessed_gbm/PM336.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(raw_documents)
# Create a vector store from the documents
db = Chroma.from_documents(documents, cohere_embeddings)

# Create Cohere's reranker with the vector DB using Cohere's embeddings as the base retriever
cohere_rerank = CohereRerank(cohere_api_key="{API_KEY}")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=cohere_rerank,
    base_retriever=db.as_retriever()
)
compressed_docs = compression_retriever.get_relevant_documents(user_query)
# Print the relevant documents from using the embeddings and reranker
print(compressed_docs)

# Create the cohere rag retriever using the chat model
rag = CohereRagRetriever(llm=cohere_chat_model)
docs = rag.get_relevant_documents(
    user_query,
    source_documents=compressed_docs,
)
# Print the documents
for doc in docs[:-1]:
    print(doc.metadata)
    print("\n\n" + doc.page_content)
    print("\n\n" + "-" * 30 + "\n\n")
# Print the final generation
answer = docs[-1].page_content
print(answer)
# Print the final citations
citations = docs[-1].metadata['citations']
print(citations)
