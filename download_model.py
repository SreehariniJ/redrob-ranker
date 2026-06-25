import os
from sentence_transformers import SentenceTransformer

def download_model():
    print("Downloading all-MiniLM-L6-v2 to local directory './models'...")
    model_dir = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
    os.makedirs(model_dir, exist_ok=True)
    # Download and save the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.save(model_dir)
    print(f"Model saved to {model_dir}")

if __name__ == "__main__":
    download_model()
