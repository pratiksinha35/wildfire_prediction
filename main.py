import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from swin_encoder_decoder_standalone import SwinEncoderDecoderTransformer3D
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------
# PARAMETERS
# -----------------------
NDVI_FOLDER = "ndvi_geotiff"  # <-- CHANGE THIS
WINDOW_SIZE = 5
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# STEP 1: Load NDVI images
# -----------------------
def load_ndvi_series(folder):
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".tif")])
    ndvi_images = []

    for f in files:
        with rasterio.open(f) as src:
            ndvi = src.read(1)  # single-band NDVI
            ndvi_resized = resize(ndvi, (256, 256), anti_aliasing=True, preserve_range=True)
            ndvi_images.append(ndvi_resized)

    stack = np.stack(ndvi_images, axis=0)  # (T, H, W)
    return stack


# -----------------------
# STEP 2: Normalize to [0, 1]
# -----------------------
def normalize_ndvi(ndvi_stack):
    return (ndvi_stack + 1.0) / 2.0


# -----------------------
# STEP 3: Create input/output windows
# -----------------------
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    X = np.array(X)[:, np.newaxis, :, :, :]  # (samples, 1, time, H, W)
    y = np.array(y)[:, np.newaxis, :, :]     # (samples, 1, H, W)
    return X, y


# -----------------------
# STEP 4: Dataset class
# -----------------------
class NDVIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



# -----------------------
# STEP 6: Training loop
# -----------------------
def train():
    print("Loading data...")
    ndvi_stack = load_ndvi_series(NDVI_FOLDER)
    ndvi_stack = normalize_ndvi(ndvi_stack)
    X, y = create_sequences(ndvi_stack, WINDOW_SIZE)

    dataset = NDVIDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SwinEncoderDecoderTransformer3D(in_chans=1, n_divs=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("Training...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    # Optional: Save model
    torch.save(model.state_dict(), "ndvi_model.pt")
    print("Model saved to ndvi_model.pt")

    # Optional: visualize last prediction
    model.eval()
    with torch.no_grad():
        test_X, test_y = dataset[-1]
        test_X = test_X.unsqueeze(0).to(DEVICE)  # add batch dim
        pred = model(test_X).cpu().squeeze().numpy()
        target = test_y.squeeze().numpy()

        plt.subplot(1, 2, 1)
        plt.imshow(target, cmap="viridis")
        plt.title("Target NDVI")

        plt.subplot(1, 2, 2)
        plt.imshow(pred, cmap="viridis")
        plt.title("Predicted NDVI")

        plt.show()


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    train()
