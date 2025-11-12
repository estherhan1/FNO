import torch
import matplotlib.pyplot as plt
import numpy as np

# 載入資料
data = torch.load('Data.pt')
train_in  = data['train_in']
train_sol = data['train_sol']

print("Data loaded:")
print("train_in :", train_in.shape)
print("train_sol:", train_sol.shape)

# 隨機選幾筆樣本
indices = [0, 100, 500, 1000]   # 你也可以改成 random.sample(range(len(train_in)), 4)
n = len(indices)

# 嘗試不同色圖（你可以改成 'coolwarm', 'viridis', 'magma', 'RdBu', 'seismic'）
cmaps = ['coolwarm', 'viridis', 'magma', 'seismic']

for cmap in cmaps:
    fig, axes = plt.subplots(nrows=n, ncols=4, figsize=(12, 3*n))
    for i, idx in enumerate(indices):
        x_in, y_in = train_in[idx]
        x_out, y_out = train_sol[idx]

        # 輸入 x, 輸入 y, 輸出 x, 輸出 y
        for j, (arr, title) in enumerate([
            (x_in, 'Input X'),
            (y_in, 'Input Y'),
            (x_out, 'Output X'),
            (y_out, 'Output Y')
        ]):
            ax = axes[i, j]
            im = ax.imshow(arr.numpy(), cmap=cmap)
            ax.set_title(f"Sample {idx} - {title}", fontsize=8)
            ax.axis('off')
    plt.suptitle(f"Colormap = {cmap}", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(f"visualize_samples_{cmap}.png", dpi=150)
    plt.close()

print("✅ Visualization images saved as visualize_samples_*.png")

import torch
import matplotlib.pyplot as plt
import numpy as np

# 假設你的資料已經載入
# train_in: (N, 2, 64, 64)
# train_sol: (N, 2, 64, 64)

samples = [0, 100, 500, 1000]  # 想要展示的樣本編號
n = len(samples)

plt.figure(figsize=(12, 3 * n))

for i, idx in enumerate(samples):
    u_in, v_in = train_in[idx, 0].numpy(), train_in[idx, 1].numpy()
    u_out, v_out = train_sol[idx, 0].numpy(), train_sol[idx, 1].numpy()

    X, Y = np.meshgrid(np.arange(u_in.shape[1]), np.arange(u_in.shape[0]))

    # Input 向量場
    plt.subplot(n, 4, 4 * i + 1)
    plt.quiver(X, Y, u_in, v_in, scale=50, cmap='coolwarm')
    plt.title(f"Sample {idx} - Input Field")
    plt.axis('off')

    # Output 向量場
    plt.subplot(n, 4, 4 * i + 2)
    plt.quiver(X, Y, u_out, v_out, scale=50, cmap='coolwarm')
    plt.title(f"Sample {idx} - Output Field")
    plt.axis('off')

    # Input magnitude
    plt.subplot(n, 4, 4 * i + 3)
    mag_in = np.sqrt(u_in**2 + v_in**2)
    plt.imshow(mag_in, cmap='viridis')
    plt.title(f"Sample {idx} - |Input|")
    plt.axis('off')

    # Output magnitude
    plt.subplot(n, 4, 4 * i + 4)
    mag_out = np.sqrt(u_out**2 + v_out**2)
    plt.imshow(mag_out, cmap='viridis')
    plt.title(f"Sample {idx} - |Output|")
    plt.axis('off')

plt.tight_layout()
plt.savefig("vector_field_quiver.png", dpi=300)
plt.show()

