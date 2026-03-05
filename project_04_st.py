import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import faiss
import os
import matplotlib.pyplot as plt

# -----------------------------
# Load FAISS index + embeddings
# -----------------------------
EMB_PATH = "./project_04/data/cars_embeddings.npy"
NAME_PATH = "./project_04/data/cars_filenames.npy"
LABEL_PATH = "./project_04/data/cars_labels.npy"

embeddings = np.load(EMB_PATH).astype('float32')
filenames = np.load(NAME_PATH)
labels = np.load(LABEL_PATH)

# Normalize embeddings
faiss.normalize_L2(embeddings)

# Build FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# -----------------------------
# Load ResNet50 model
# -----------------------------
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_embedding(img: Image.Image):
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(tensor).squeeze().numpy().astype('float32')
    faiss.normalize_L2(emb.reshape(1, -1))
    return emb.reshape(1, -1)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚗 Car Image Similarity Search")
st.write("Upload a car image and find visually similar cars from the dataset.")

uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    query_img = Image.open(uploaded_file).convert("RGB")
    st.image(query_img, caption="Query Image", use_container_width=True)

    # Get embedding
    query_emb = get_embedding(query_img)

    # Search FAISS index
    k = st.slider("Number of similar cars to show", 1, 30, 5)
    distances, indices = index.search(query_emb, k)

    st.subheader("Top Similar Cars")
    cols = st.columns(k)
    for i, idx in enumerate(indices[0]):
        img_path = os.path.join("./project_04/data/cars_subset", filenames[idx])
        sim_img = Image.open(img_path)
        with cols[i]:
            st.image(sim_img, caption=f"{filenames[idx]} (dist={distances[0][i]:.4f})")

# -----------------------------
# Define evaluation function
# -----------------------------
def evaluate_precision_recall(k=30, num_queries=50):
    precisions = []
    recalls = []

    # Randomly pick some queries
    query_indices = np.random.choice(len(embeddings), num_queries, replace=False)

    for qi in query_indices:
        query_emb = embeddings[qi].reshape(1, -1)
        query_label = labels[qi]

        distances, indices = index.search(query_emb, k)

        # Count relevant results
        retrieved_labels = labels[indices[0]]
        relevant = np.sum(retrieved_labels == query_label)

        # Precision@K
        precision = relevant / k
        precisions.append(precision)

        # Recall@K
        total_relevant = np.sum(labels == query_label)
        recall = relevant / total_relevant if total_relevant > 0 else 0
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls)

# -----------------------------
# Run evaluation
# -----------------------------
if st.button("Run Evaluation"):
    precision, recall = evaluate_precision_recall(k=5, num_queries=50)
    st.write(f"Precision@5: {precision:.3f}")
    st.write(f"Recall@5: {recall:.3f}")

    # Run evaluation for multiple K values
    K_values = [5, 10, 20, 30, 50, 100]
    precisions, recalls = [], []

    for k in K_values:
        p, r = evaluate_precision_recall(k=k, num_queries=50)
        precisions.append(p)
        recalls.append(r)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(K_values, precisions, marker='o', label="Precision@K")
    ax.plot(K_values, recalls, marker='s', label="Recall@K")
    ax.set_xlabel("K (Number of retrieved results)")
    ax.set_ylabel("Score")
    ax.set_title("Precision vs Recall Trade-off")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig) 

