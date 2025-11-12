import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from neuralop.models import FNO
from neuralop.losses import LpLoss
import wandb  #
import numpy as np

# ==========================================
# 1. 設定與數據載入 (Setup & Data Loading)
# ==========================================

# 設定裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 載入數據
try:
    data = torch.load('Data.pt')
except FileNotFoundError:
    print("錯誤: 找不到 Data.pt，請確認檔案路徑。")
    exit()

train_in  = data['train_in']
train_sol = data['train_sol']
test_in   = data['test_in']
test_sol  = data['test_sol']

print(f"Train data shape: {train_in.shape}")
print(f"Test data shape: {test_in.shape}")

# 建立 DataLoader
batch_size = 32
train_dataset = TensorDataset(train_in, train_sol)
test_dataset  = TensorDataset(test_in, test_sol)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==========================================
# 2. 初始化 WandB 與 模型 (WandB & Model)
# ==========================================

# 超參數
"""modes = 16
width = 64
learning_rate = 1e-3
epochs = 50
scheduler_step = 10
scheduler_gamma = 0.5"""
modes = 32           # 增加頻率特徵
width = 64           # 維持或加到 96
learning_rate = 1e-3 # 維持
epochs = 100         # 延長訓練
scheduler_step = 20  # 延後衰減
scheduler_gamma = 0.7 # 減緩衰減幅度

# 初始化 WandB
wandb.init(
    project="FNO-Kolmogorov",
    config={
        "architecture": "FNO",
        "modes": modes,
        "width": width,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "scheduler_step": scheduler_step,
        "scheduler_gamma": scheduler_gamma,
        "dataset": "Kolmogorov Flow"
    }
)

# 建立模型
model = FNO(n_modes=(modes, modes),
            hidden_channels=width,
            in_channels=2,
            out_channels=2).to(device)

# 計算並記錄參數數量
param_count = sum(p.numel() for p in model.parameters())
print(f"Model Parameters: {param_count}")
wandb.log({"model_parameters": param_count})

# Loss 與 Optimizer
criterion = LpLoss(d=2, p=2)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

# ==========================================
# 3. 訓練迴圈 (Training Loop)
# ==========================================

train_losses, test_losses = [], []

print("Starting training...")
for epoch in range(epochs):
    model.train()
    train_err = 0.0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        train_err += loss.item()
    
    train_err /= len(train_loader)
    train_losses.append(train_err)
    
    # Validation
    model.eval()
    test_err = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_err += criterion(out, y).item()
            
    test_err /= len(test_loader)
    test_losses.append(test_err)
    
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    
    # 記錄到 WandB
    wandb.log({
        "train_loss": train_err,
        "test_loss": test_err,
        "learning_rate": current_lr,
        "epoch": epoch + 1
    })
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_err:.5f} | Test Loss: {test_err:.5f}")

# 儲存模型
torch.save(model.state_dict(), "model_fno.pt")
print("Training finished and model saved.")

# ==========================================
# 4. 視覺化功能 (Visualization - Task 1 & 3)
# ==========================================

def visualize_results(model, loader, device, title_suffix=""):
    """
    隨機取樣並畫出 Input(x), Ground Truth(y), Prediction(y_hat)
    """
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(loader))
        inputs, targets = inputs.to(device), targets.to(device)
        preds = model(inputs)
    
    # 取第一個樣本進行繪圖
    idx = 0
    in_field = inputs[idx].cpu().numpy()   # (2, 64, 64)
    gt_field = targets[idx].cpu().numpy()  # (2, 64, 64)
    pred_field = preds[idx].cpu().numpy()  # (2, 64, 64)
    
    # 畫圖: 3 rows (Input, Target, Pred), 2 cols (X-component, Y-component)
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle(f'Vector Fields Visualization {title_suffix}', fontsize=16)

    # Row 1: Input
    axs[0, 0].imshow(in_field[0], cmap='RdBu_r')
    axs[0, 0].set_title('Input (X-comp)')
    axs[0, 1].imshow(in_field[1], cmap='RdBu_r')
    axs[0, 1].set_title('Input (Y-comp)')
    
    # Row 2: Ground Truth
    axs[1, 0].imshow(gt_field[0], cmap='viridis') #
    axs[1, 0].set_title('Ground Truth (X-comp)')
    axs[1, 1].imshow(gt_field[1], cmap='viridis')
    axs[1, 1].set_title('Ground Truth (Y-comp)')
    
    # Row 3: Prediction
    axs[2, 0].imshow(pred_field[0], cmap='viridis')
    axs[2, 0].set_title('Prediction (X-comp)')
    axs[2, 1].imshow(pred_field[1], cmap='viridis')
    axs[2, 1].set_title('Prediction (Y-comp)')
    
    plt.tight_layout()
    # 上傳圖片到 WandB
    wandb.log({f"visualization_{title_suffix}": wandb.Image(plt)})
    plt.show()

print("\nGenerating Visualization...")
visualize_results(model, test_loader, device, title_suffix="(64x64)")

# ==========================================
# 5. 解析度測試 (Resolution Tests - Task 6)
# ==========================================
# 測試模型在不同解析度下的 Zero-shot 能力

def test_resolution(model, data_in, data_sol, target_res, device):
    print(f"\nRunning Resolution Test: {target_res}x{target_res} ...")
    
    # 1. 將輸入資料插值到目標解析度
    # input shape: (N, 2, 64, 64) -> (N, 2, target_res, target_res)
    x_res = F.interpolate(data_in, size=(target_res, target_res), mode='bicubic', align_corners=False)
    y_res = F.interpolate(data_sol, size=(target_res, target_res), mode='bicubic', align_corners=False)
    
    dataset_res = TensorDataset(x_res, y_res)
    loader_res = DataLoader(dataset_res, batch_size=32, shuffle=False)
    
    model.eval()
    total_loss = 0.0
    criterion_res = LpLoss(d=2, p=2)
    
    with torch.no_grad():
        for x, y in loader_res:
            x, y = x.to(device), y.to(device)
            
            # FNO 可以接受任意解析度的輸入
            out = model(x)
            
            loss = criterion_res(out, y)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(loader_res)
    print(f"Resolution {target_res}x{target_res} - Test Loss: {avg_loss:.5f}")
    wandb.log({f"loss_{target_res}x{target_res}": avg_loss})
    
    # 視覺化其中一張圖
    visualize_results(model, loader_res, device, title_suffix=f"({target_res}x{target_res})")

# 測試 32x32 (Downsampling)
test_resolution(model, test_in, test_sol, 32, device)

# 測試 128x128 (Upsampling)
# 注意：這裡使用插值後的 GT 作為參考，主要是觀察視覺效果
test_resolution(model, test_in, test_sol, 128, device)

# 結束 WandB
wandb.finish()