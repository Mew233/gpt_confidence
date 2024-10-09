import PyPDF2
import cohere
import openai
import numpy as np


# Step 1: Reading PDFs with PyPDF2
def read_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


# Step 2: Use Cohere to rerank relevant texts
def rerank_texts(api_key, query, documents):
    co = cohere.Client(api_key)
    reranked = co.rerank(
        query=query,
        documents=documents,
        top_n=5  # Adjust based on your need
    )
    return [doc.text for doc in reranked.results]


# Step 3: Extract entities using ChatGPT
def extract_entities(openai_api_key, text):
    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a medical entity extractor."},
            {"role": "user", "content": f"Extract clinical entities from the following text: {text}"}
        ]
    )
    entities = response.choices[0].message['content']
    return entities


# Step 4: Classification based on confidence scores
def classify_entities_with_logits(entities):
    # Simulating a logits confidence score for each entity extracted (0 to 1)
    entity_classifications = {}
    for entity in entities:
        confidence_score = np.random.uniform(0, 1)  # Random confidence score for example
        entity_classifications[entity] = "high" if confidence_score > 0.7 else "low"
    return entity_classifications


# Main function to integrate all steps
def main():
    # Example patient PDFs
    pdf_files = [
        '../data/PM200/PM200_SURGPATH_07122004_Craniotomy.pdf',
        '../data/PM200/PM200_SURGPATH_07232015_GBMdx.pdf',
        '../data/PM200/PM200_Encounter_07122015_OncHx.pdf',
        '../data/PM200/PM200_SURGPATH_05052014_Recurrence.pdf',
        '../data/PM200/PM200_MRI_07132008_POD.pdf'
    ]

    # 1. Extract texts from PDFs
    pdf_texts = [read_pdf(pdf) for pdf in pdf_files]

    # 2. Rerank texts with Cohere (assuming Cohere API key is set)
    cohere_api_key = 'your-cohere-api-key'
    query = "Patient history, impression, assessment, plan"
    reranked_texts = rerank_texts(cohere_api_key, query, pdf_texts)

    # 3. Extract clinical entities using ChatGPT (assuming OpenAI API key is set)
    openai_api_key = 'your-openai-api-key'
    extracted_entities = [extract_entities(openai_api_key, text) for text in reranked_texts]

    # 4. Classify entities based on logits/confidence scores
    classified_entities = [classify_entities_with_logits(entities) for entities in extracted_entities]

    print("Classified Entities with Confidence Scores:")
    for idx, classification in enumerate(classified_entities):
        print(f"Text {idx + 1}: {classification}")


if __name__ == "__main__":
    main()
