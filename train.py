import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from neuralop.models import FNO
from neuralop.losses import LpLoss


data = torch.load('Data.pt')
train_in  = data['train_in']
train_sol = data['train_sol']
test_in = data['test_in']
test_sol = data['test_sol']

print("Data loaded:")
print("train_in :", train_in.shape)
print("train_sol:", train_sol.shape)


batch_size = 32
train_dataset = TensorDataset(train_in, train_sol)
test_dataset = TensorDataset(test_in, test_sol)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)


modes = 16          # Fourier modes per dimension
width = 64          # Hidden channel width
model = FNO(n_modes=(modes, modes),
            hidden_channels=width,
            in_channels=2,
            out_channels=2).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


criterion = LpLoss(d=2, p=2)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


num_epochs = 50
train_losses, test_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # ===== Validation =====
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_loss += criterion(pred, y).item()
    val_loss /= len(test_loader)
    test_losses.append(val_loss)
    scheduler.step()

    print(f"[Epoch {epoch+1:02d}] Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")


torch.save(model.state_dict(), "fno_kolmogorov.pt")
print("✅ Model saved to fno_kolmogorov.pt")

plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Relative L2 Loss')
plt.legend()
plt.title('FNO Training Curve')
plt.grid(True)
plt.savefig('fno_training_curve.png', dpi=300)
plt.show()
