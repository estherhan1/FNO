import torch
import matplotlib.pyplot as plt
import numpy as np
from neuralop.models import FNO
from torch.utils.data import TensorDataset, DataLoader

# 1. 設定裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. 載入數據
try:
    # 同樣加上 weights_only=False 以防萬一
    data = torch.load('Data.pt', weights_only=False)
    test_in = data['test_in']
    test_sol = data['test_sol']
except Exception as e:
    print(f"❌ 載入 Data.pt 失敗: {e}")
    exit()

# 3. 載入模型
modes = 32  # 確保這跟訓練時一樣
width = 64
model = FNO(n_modes=(modes, modes), hidden_channels=width, in_channels=2, out_channels=2).to(device)

print("嘗試載入模型權重...")
try:
    # 【修正點 1】優先嘗試 model_fno.pt
    try:
        checkpoint = torch.load("model_fno.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)
        print("✅ 成功載入 model_fno.pt")
    except FileNotFoundError:
        # 【修正點 2】如果找不到，嘗試 fno_kolmogorov.pt
        print("找不到 model_fno.pt，嘗試載入 fno_kolmogorov.pt...")
        checkpoint = torch.load("fno_kolmogorov.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)
        print("✅ 成功載入 fno_kolmogorov.pt")
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")
    print("請確認您目錄下有 .pt 檔案，且訓練參數 (modes/width) 與程式碼一致。")
    exit()

# 4. 繪圖並存檔
print("正在繪製誤差圖...")
model.eval()
test_loader = DataLoader(TensorDataset(test_in, test_sol), batch_size=1, shuffle=False)

with torch.no_grad():
    inputs, targets = next(iter(test_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    preds = model(inputs)
    errors = torch.abs(targets - preds)

# 轉 CPU
idx = 0
gt = targets[idx].cpu().numpy()
pred = preds[idx].cpu().numpy()
err = errors[idx].cpu().numpy()

# 畫圖
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
rows = ['Ground Truth', 'Prediction', 'Abs Error']
cols = ['X', 'Y', 'Magnitude']
data_list = [gt, pred, err]

for i, field in enumerate(data_list):
    mag = np.sqrt(field[0]**2 + field[1]**2)
    for j, img in enumerate([field[0], field[1], mag]):
        ax = axs[i, j]
        im = ax.imshow(img, cmap='magma' if i==2 else 'viridis')
        ax.set_title(f"{rows[i]} - {cols[j]}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.savefig('error_map.png')
print("🎉 圖片已儲存為 error_map.png")