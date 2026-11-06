
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True
)

# Load FAISS vector index
index = faiss.read_index(
    "vectorstore/faiss_index.bin"
)

# Load stored text chunks
with open(
    "vectorstore/chunks.pkl",
    "rb"
) as f:
    chunks = pickle.load(f)

# Get question from user
query = input("Ask Question: ")

# Convert question into embedding
query_embedding = model.encode([query])

query_embedding = np.array(
    query_embedding,
    dtype=np.float32
)

# Normalize embedding
faiss.normalize_L2(query_embedding)

# Search top 3 similar results
distances, indices = index.search(
    query_embedding,
    3
)

# Display retrieved results
for rank, idx in enumerate(indices[0], start=1):
    if idx == -1:
        continue

    print("\n")
    print(f"RESULT {rank}")
    print("=" * 60)

    print(chunks[idx])

    print("\nDistance:", distances[0][rank - 1])
    print()
