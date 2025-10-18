import json
import os
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Load product data
with open('data/products.json', 'r') as f:
    products = json.load(f)

# Create directories for FAISS indexes and metadata
os.makedirs('vector_stores/image_faiss', exist_ok=True)
os.makedirs('vector_stores/text_faiss', exist_ok=True)

# --- Image Indexing (Manual) ---

# Load CLIP model
image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_embeddings = []
image_metadata = []

for product in products:
    image_path = product['image_path']
    if os.path.exists(image_path):
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = image_model.get_image_features(**inputs)
        image_embeddings.append(embedding.cpu().numpy().flatten())
        image_metadata.append(product)

# Create and save image FAISS index
if image_embeddings:
    image_embeddings = np.array(image_embeddings).astype('float32')
    d = image_embeddings.shape[1]
    image_index = faiss.IndexFlatL2(d)
    image_index.add(image_embeddings)
    faiss.write_index(image_index, 'vector_stores/image_faiss/image.index')

    # Save image metadata
    with open('vector_stores/image_faiss/image_metadata.json', 'w') as f:
        json.dump(image_metadata, f)

# --- Text Indexing (LangChain) ---

# Create documents for LangChain
documents = []
for product in products:
    product_text = f"Name: {product['name']}\nDescription: {product['description']}\nPrice: {product['price']}\nMarginal Price: {product['marginal_price']}\nCode: {product['code']}\nLink: {product['link']}"
    documents.append(Document(page_content=product_text, metadata=product))

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

# Create and save text FAISS vector store
vector_store = FAISS.from_documents(documents, embeddings)
vector_store.save_local('vector_stores/text_faiss')

print("Training complete.")