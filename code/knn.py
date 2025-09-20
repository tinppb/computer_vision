import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Đường dẫn dataset
data_dir = "D:\\HUET\\NAM 4\\NAM 4 KI I\\computer_vision\\datasets\\ShoeSandalBoot\\Shoe vs Sandal vs Boot Dataset"

# Các lớp
classes = ["Shoe", "Sandal", "Boot"]

X = []  # đặc trưng
y = []  # nhãn

# Đọc ảnh từ thư mục
for label, class_name in enumerate(classes):
    folder = os.path.join(data_dir, class_name)
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # đọc ảnh grayscale
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))  # resize về 64x64
        X.append(img.flatten())          # chuyển thành vector 1D
        y.append(label)

X = np.array(X)
y = np.array(y)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Huấn luyện KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=classes))

# ==============================
# Hàm dự đoán ảnh mới
# ==============================
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Không đọc được ảnh:", image_path)
        return
    img = cv2.resize(img, (64, 64))
    img_flat = img.flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_flat)  # chuẩn hóa theo scaler đã fit
    prediction = knn.predict(img_scaled)[0]
    print(f"Ảnh {image_path} được dự đoán là: {classes[prediction]}")

# Thử dự đoán một ảnh mới
predict_image("D:\\HUET\\NAM 4\\NAM 4 KI I\\computer_vision\\datasets\\t0.jpg")
