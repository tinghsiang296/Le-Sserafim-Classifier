import torch
# Force disable MPS
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from main import download_dataset, train_model

if __name__ == '__main__':
    # print("Starting download...")
    # download_dataset()
    print("Starting training...")
    learn, dls = train_model()
    print("Exporting model...")
    learn.export('le_sserafim_model.pkl')
    print("Model exported to le_sserafim_model.pkl")
