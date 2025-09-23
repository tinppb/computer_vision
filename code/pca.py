import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import pandas as pd
# ======================
# 1. Chuẩn bị dữ liệu
# ======================
data_dir = r"D:\datasets\ShoeSandalBoot"
classes = ["Shoe", "Sandal", "Boot"]

IMG_SIZE = 90  # resize ảnh kích thước 120x120

X = []
y = []

img_size = (64, 64)

for label, c in enumerate(classes):
    folder = os.path.join(data_dir, c)
    for img_file in os.listdir(folder)[:200]:  # lấy 50 ảnh cho nhẹ
        img = load_img(os.path.join(folder, img_file), target_size=img_size)
        arr = img_to_array(img).flatten() / 255.0
        X.append(arr)
        y.append(c)  # dùng tên class thay vì số cho dễ đọc

X = np.array(X)
# #PCA
# pca = PCA(n_components=2)
# X_2d = pca.fit_transform(X)

# df = pd.DataFrame(X_2d, columns=["Feature 1", "Feature 2"])
# df["Target"] = y

# plt.figure(figsize=(8, 6))
# sns.scatterplot(
#     data=df,
#     x="Feature 1", y="Feature 2",
#     hue="Target", palette="tab10", legend="full", s=70
# )
# plt.title("Shoes vs Sandal vs Boot (PCA 2D Projection)")
# plt.grid(True)
# plt.show()

pca = PCA(n_components=3)
X_3d = pca.fit_transform(X)

df = pd.DataFrame(X_3d, columns=["Feature 1", "Feature 2", "Feature 3"])
df["Target"] = y

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for target in df["Target"].unique():
    subset = df[df["Target"] == target]
    ax.scatter(subset["Feature 1"], subset["Feature 2"], subset["Feature 3"], label=target, s=50)

ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
ax.set_title("Shoes vs Sandal vs Boot (PCA 3D Projection)")
ax.legend()
plt.show()
