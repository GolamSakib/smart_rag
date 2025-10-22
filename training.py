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
import requests
from ultralytics import YOLO

# Load product data
with open('data/products.json', 'r') as f:
    products = json.load(f)

# Create directories for FAISS indexes and metadata
os.makedirs('vector_stores/image_faiss', exist_ok=True)
os.makedirs('vector_stores/text_faiss', exist_ok=True)

# --- Image Indexing with YOLO Object Detection ---

# Load YOLO object detector
object_detector = YOLO("yolov8n.pt")
TARGET_CLASSES = {24, 26, 28}  # backpack (24), handbag (26), suitcase (28)

# Load CLIP model
image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_embeddings = []
image_metadata = []

# Statistics tracking
stats = {
    "total_images": 0,
    "object_detected": 0,
    "fallback_used": 0,
    "skipped": 0
}

for product in products:
    # Handle both single image_path and multiple image_paths
    image_paths = product.get("image_paths", [product.get("image_path")])
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    for image_path in image_paths:
        if os.path.exists(image_path):
            stats["total_images"] += 1
            image = Image.open(image_path).convert("RGB")
            target_object_found = False
            
            # Detect objects using YOLO
            results = object_detector(image)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and boxes.shape[0] > 0:  # If objects detected
                    for i in range(boxes.shape[0]):
                        cls = boxes.cls[i].item()  # Class ID
                        if cls in TARGET_CLASSES:  # Filter for handbag (23) or shoe (40)
                            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                            cropped_image = image.crop((x1, y1, x2, y2))
                            
                            # Generate embedding for cropped object
                            inputs = processor(images=cropped_image, return_tensors="pt")
                            with torch.no_grad():
                                embedding = image_model.get_image_features(**inputs)
                            image_embeddings.append(embedding.cpu().numpy().flatten())
                            image_metadata.append(product)
                            target_object_found = True
                            stats["object_detected"] += 1
                            print(f"✅ Processed {image_path} with detected object (class {cls})")
                            break
                    if target_object_found:
                        break
            
            # If no target objects detected, use the entire image as fallback
            if not target_object_found:
                print(f" No target objects detected in {image_path}, using full image")
                inputs = processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    embedding = image_model.get_image_features(**inputs)
                image_embeddings.append(embedding.cpu().numpy().flatten())
                image_metadata.append(product)
                stats["fallback_used"] += 1
        else:
            stats["skipped"] += 1
            print(f"Image not found: {image_path}")

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

# Print statistics
print("\n" + "="*50)
print("TRAINING STATISTICS:")
print(f"Total images processed: {stats['total_images']}")
print(f"Images with object detection: {stats['object_detected']} ({stats['object_detected']/max(stats['total_images'],1)*100:.1f}%)")
print(f"Images using fallback (full image): {stats['fallback_used']} ({stats['fallback_used']/max(stats['total_images'],1)*100:.1f}%)")
print(f"Images skipped (not found): {stats['skipped']}")
print(f"Total embeddings created: {len(image_embeddings)}")
print("="*50)

# --- Reload Models ---
try:
    # response = requests.post("http://127.0.0.1:8000/api/reload-models")
    response = requests.post("https://chat.momsandkidsworld.com/api/reload-models")
    if response.status_code == 200:
        print("Models reloaded successfully.")
    else:
        print(f"Failed to reload models. Status code: {response.status_code}")
except requests.exceptions.ConnectionError as e:
    print(f"Failed to connect to the server: {e}")
    print("\nTo load the new data into the running application, send a POST request to the /api/reload-models endpoint.")