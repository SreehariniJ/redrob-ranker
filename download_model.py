import os
from sentence_transformers import SentenceTransformer

def download_model():
    print("Downloading BAAI/bge-small-en-v1.5 to local directory './models'...")
    model_dir = os.path.join(os.path.dirname(__file__), "models", "bge-small-en-v1.5")
    os.makedirs(model_dir, exist_ok=True)
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    model.save(model_dir)
    print(f"Model saved to {model_dir}")

if __name__ == "__main__":
    download_model()
