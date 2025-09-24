import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# ======================
# 1. Cấu hình
# ======================
img_size = (64, 64)   # Resize ảnh về 64x64, bạn có thể đổi (20,20) hay (120,120)
k = 3

# ======================
# 2. Hàm load ảnh
# ======================
def load_images_from_folder(folder, label):
    data, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # đọc ảnh Gray
        if img is None:
            continue
        img = cv2.resize(img, img_size)                  # resize
        img_flat = img.flatten() / 255.0                 # chuẩn hóa [0,1]
        data.append(img_flat)
        labels.append(label)
    return data, labels

# ======================
# 3. Load dataset + lưu tên class
# ======================
X, y = [], []
class_names = []  # để map từ số -> tên class

root_dir = r"D:\datasets\ShoeSandalBoot"   # thư mục train (gồm sub-folder: sandal, boot, shoe,...)

for label, class_name in enumerate(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, class_name)
    if not os.path.isdir(folder_path):
        continue
    data, labels = load_images_from_folder(folder_path, label)
    X.extend(data)
    y.extend(labels)
    class_names.append(class_name)   # lưu lại tên class

X = np.array(X)
y = np.array(y)

print("Class names:", class_names)   # VD: ['boot', 'sandal', 'shoe']

# ======================
# 4. Chia train/test
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ======================
# 5. Train KNN
# ======================
knn_k = KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="distance")
knn_k.fit(X_train_scaled, y_train)

print("Train accuracy:", knn_k.score(X_train_scaled, y_train))
print("Test accuracy :", knn_k.score(X_test_scaled, y_test))

# ======================
# 6. Hàm dự đoán folder ảnh
# ======================
def predict_folder(folder_path, limit=None):
    results = []
    count = 0
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        img_flat = img.flatten().reshape(1, -1) / 255.0
        img_scaled = scaler.transform(img_flat)
        pred_label = knn_k.predict(img_scaled)[0]
        pred_name = class_names[pred_label]   # map số -> tên

        results.append((filename, pred_name))

        plt.imshow(img, cmap="gray")
        plt.title(f"{filename}\nDự đoán: {pred_name}")
        plt.axis("off")
        plt.show()

        print(f"{filename} → {pred_name}")

        count += 1
        if limit and count >= limit:
            break
    
    return results

# ======================
# 7. Test predict folder
# ======================
test_dir = r"D:\datasets\test" 
results = predict_folder(test_dir, limit=16)  # hiển thị thử 16 ảnh đầu
