# app/system.py

import os
import json
import warnings
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# 1. Graph Visualizer
# ---------------------------------------------------------------------
class GraphVisualizer:
    def __init__(self):
        self.graph = nx.Graph()

    def build_graph(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            user = row["user_id"]
            self.graph.add_node(user)
            for col in df.columns:
                if col not in ["user_id"]:
                    self.graph.add_edge(user, f"{col}:{row[col]}")
        return self.graph

    def get_user_subgraph(self, user_id):
        neighbors = list(self.graph.neighbors(user_id))
        subgraph_nodes = [user_id] + neighbors
        return self.graph.subgraph(subgraph_nodes)

# ---------------------------------------------------------------------
# 2. Lightweight Vector Store
# ---------------------------------------------------------------------
class LightweightVectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.store = {}

    def add(self, key: str, text: str):
        embedding = self.encoder.encode(text)
        self.store[key] = {"text": text, "embedding": embedding}

    def query(self, query_text: str, top_k: int = 3):
        if not self.store:
            return []
        query_vec = self.encoder.encode(query_text)
        keys = list(self.store.keys())
        embeddings = np.array([self.store[k]["embedding"] for k in keys])
        sims = cosine_similarity([query_vec], embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = [(keys[i], sims[i], self.store[keys[i]]["text"]) for i in top_indices]
        return results

# ---------------------------------------------------------------------
# 3. Medical Knowledge Graph
# ---------------------------------------------------------------------
class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_relationship(self, node1, relation, node2):
        self.graph.add_edge(node1, node2, relation=relation)

    def query(self, node):
        if node not in self.graph:
            return []
        return list(self.graph[node].items())

# ---------------------------------------------------------------------
# 4. Graph-Enhanced Dataset
# ---------------------------------------------------------------------
class GraphEnhancedDataset(torch.utils.data.Dataset):
    def __init__(self, df, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
        self.df = df

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------------------------------------------------
# 5. Knowledge-Enhanced Graph Transformer
# ---------------------------------------------------------------------
class KnowledgeEnhancedGraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        attn_output, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = attn_output.squeeze(1)
        x = self.fc2(x)
        return x

# ---------------------------------------------------------------------
# 6. Model Trainer
# ---------------------------------------------------------------------
class ModelTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, dataloader, num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for X, y in dataloader:
                X, y = X.to(self.device), y.long().to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
        print("Classification Report:\n", classification_report(all_labels, all_preds))

# ---------------------------------------------------------------------
# 7. Real LLM Client (e.g., OpenAI)
# ---------------------------------------------------------------------
class RealLLMClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_explanation(self, prompt: str) -> str:
        """
        Replace with actual OpenAI API call if deployed with key.
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print("⚠️ LLM call failed:", e)
            return "LLM explanation unavailable at the moment."

# ---------------------------------------------------------------------
# 8. Real LLM RAG System
# ---------------------------------------------------------------------
class RealLLMRAGSystem:
    def __init__(self, vector_store, llm_client):
        self.vector_store = vector_store
        self.llm_client = llm_client

    def retrieve_and_generate(self, query: str) -> str:
        results = self.vector_store.query(query)
        if not results:
            return "No relevant data found."
        context = "\n".join([r[2] for r in results])
        prompt = (
            f"Use the following menstrual health knowledge to answer:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        return self.llm_client.get_explanation(prompt)

# ---------------------------------------------------------------------
# 9. Integrated System
# ---------------------------------------------------------------------
class IntegratedRealLLMSystem:
    def __init__(self, df: pd.DataFrame, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Using device: {self.device}")

        # Encode phases
        self.label_encoder = LabelEncoder()
        df["phase_label"] = self.label_encoder.fit_transform(df["phase"])

        # Standardize features
        features = df[["cycle_length", "bleeding_intensity", "bbt", "mood_score"]].values
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features)

        # Dataset & dataloaders
        dataset = GraphEnhancedDataset(df, scaled_features, df["phase_label"].values)
        train_size = int(0.8 * len(dataset))
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=16)

        # Model & trainer
        input_dim = scaled_features.shape[1]
        hidden_dim = 64
        num_classes = len(self.label_encoder.classes_)
        self.model = KnowledgeEnhancedGraphTransformer(input_dim, hidden_dim, num_classes)
        self.trainer = ModelTrainer(self.model, self.device)

        # RAG setup
        self.vector_store = LightweightVectorStore()
        self.llm_client = RealLLMClient(os.getenv("OPENAI_API_KEY", ""))
        self.rag_system = RealLLMRAGSystem(self.vector_store, self.llm_client)

        # Add some base knowledge
        self._add_default_knowledge()

        self.df = df

    def _add_default_knowledge(self):
        knowledge_texts = [
            "During the follicular phase, estrogen levels rise and follicles develop.",
            "The luteal phase follows ovulation and is dominated by progesterone.",
            "High BBT indicates ovulation has occurred.",
            "Mood changes can occur in the luteal phase due to hormonal shifts."
        ]
        for i, text in enumerate(knowledge_texts):
            self.vector_store.add(f"doc_{i}", text)

    def train_model(self, num_epochs=5):
        print("🚀 Starting training...")
        self.trainer.train(self.train_loader, num_epochs)
        print("✅ Training completed.")
        print("📊 Evaluation:")
        self.trainer.evaluate(self.test_loader)

    def predict_with_explanation(self, user_id: int) -> Optional[Dict[str, Any]]:
        user_row = self.df[self.df["user_id"] == user_id]
        if user_row.empty:
            return None

        X_user = user_row[["cycle_length", "bleeding_intensity", "bbt", "mood_score"]].values
        X_user_scaled = torch.tensor(self.scaler.transform(X_user), dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(X_user_scaled)
            probs = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        predicted_phase = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence_level = (
            "high" if confidence > 0.8 else "moderate" if confidence > 0.5 else "low"
        )

        query = (
            f"Explain the {predicted_phase} phase in relation to menstrual health, "
            f"given the features: cycle_length={user_row['cycle_length'].values[0]}, "
            f"bbt={user_row['bbt'].values[0]}, mood_score={user_row['mood_score'].values[0]}"
        )

        explanation = self.rag_system.retrieve_and_generate(query)

        return {
            "prediction": {
                "predicted_phase": predicted_phase,
                "days_to_menstrual": float(user_row["days_to_menstrual"].values[0]),
                "confidence": confidence_level,
                "confidence_score": confidence
            },
            "explanation": explanation
        }

# ---------------------------------------------------------------------
# 10. Entry point for FastAPI integration
# ---------------------------------------------------------------------
def build_system(csv_path: str, openai_api_key: Optional[str] = None, device: Optional[str] = None):
    """
    Initializes the full IntegratedRealLLMSystem using the provided dataset and API key.
    This function is called by the FastAPI app at startup.
    """
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    system = IntegratedRealLLMSystem(df, device=device)
    return system
