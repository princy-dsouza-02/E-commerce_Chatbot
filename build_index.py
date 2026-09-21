import pandas as pd
import numpy as np
import faiss
import pickle

from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/bitext_retail.csv")

print("Dataset loaded:", df.shape)

# Remove rows with missing instructions
df = df.dropna(subset=["instruction"])

# Reset index
df = df.reset_index(drop=True)

# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")

# --------------------------------------------------
# 3. Create embeddings
# --------------------------------------------------

texts = df["instruction"].tolist()

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

# Convert to float32 for FAISS
embeddings = embeddings.astype("float32")

# --------------------------------------------------
# 4. Normalize embeddings
# --------------------------------------------------

faiss.normalize_L2(embeddings)

# --------------------------------------------------
# 5. Create FAISS index
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("Number of vectors:", index.ntotal)

# --------------------------------------------------
# 6. Save FAISS index
# --------------------------------------------------

faiss.write_index(
    index,
    "models/retail_faiss.index"
)

# --------------------------------------------------
# 7. Save dataframe
# --------------------------------------------------

df.to_pickle(
    "models/retail_data.pkl"
)

print("FAISS index saved.")
print("Dataset metadata saved.")
