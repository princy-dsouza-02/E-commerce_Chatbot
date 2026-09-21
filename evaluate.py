import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
import faiss


# Load data
df = pd.read_csv("data/bitext_retail.csv")

df = df.dropna(
    subset=["instruction", "intent", "category"]
).reset_index(drop=True)


# Split data
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["intent"]
)


print("Training records:", len(train_df))
print("Testing records:", len(test_df))


# Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Encode training examples
train_embeddings = model.encode(
    train_df["instruction"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
).astype("float32")


faiss.normalize_L2(train_embeddings)


# Create FAISS index
dimension = train_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(train_embeddings)


# Encode test examples
test_embeddings = model.encode(
    test_df["instruction"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
).astype("float32")


faiss.normalize_L2(test_embeddings)


# Search
scores, indices = index.search(
    test_embeddings,
    1
)


predicted_intents = []

for idx in indices[:, 0]:

    predicted_intents.append(
        train_df.iloc[idx]["intent"]
    )


actual_intents = test_df["intent"].tolist()


# Accuracy
accuracy = accuracy_score(
    actual_intents,
    predicted_intents
)


print("\nIntent Accuracy:", accuracy)

print(
    classification_report(
        actual_intents,
        predicted_intents
    )
)

predicted_categories = []

for idx in indices[:, 0]:

    predicted_categories.append(
        train_df.iloc[idx]["category"]
    )

actual_categories = test_df["category"].tolist()

category_accuracy = accuracy_score(
    actual_categories,
    predicted_categories
)

print(
    "Category Accuracy:",
    category_accuracy
)