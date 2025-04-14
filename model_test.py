import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import random
import matplotlib.image as mpimg
import os

# Modeli yükle
model = load_model("kolondeneme/mlp_model.h5")

# Test verilerini yükle
X_test = np.load("features.npy")
Y_test = np.load("labels.npy")

# Modelden tahmin al
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5).astype(int)  # Sigmoid çıktısını 0 veya 1'e çevir

# Karışıklık Matrisi
conf_matrix = confusion_matrix(Y_test, Y_pred)

# Karışıklık Matrisini Görselleştir
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, cmap="Blues", interpolation="nearest")
plt.title("Karışıklık Matrisi")
plt.colorbar()
plt.ylabel("Gerçek Etiket")
plt.xlabel("Tahmin Edilen Etiket")
plt.xticks([0, 1], ["Kansersiz", "Kanserli"])
plt.yticks([0, 1], ["Kansersiz", "Kanserli"])

# Sayıları matrise yazdır
thresh = conf_matrix.max() / 2  # Yazı rengi kontrastı için eşik
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.show()

# Sınıflandırma Raporu
print("\nSınıflandırma Raporu:")
print(classification_report(Y_test, Y_pred, target_names=["Kansersiz", "Kanserli"]))

# Kanserli ve Kansersiz görüntülerini grupla
base_dir = "C:\\Users\\fidan\\OneDrive\\Masaüstü\\colon_image_sets"
cancerous_dir = os.path.join(base_dir, "colon_aca")
non_cancerous_dir = os.path.join(base_dir, "colon_n")

# 5 Kanserli ve 5 Kansersiz görüntü seç
cancerous_images = random.sample(os.listdir(cancerous_dir), 5)
non_cancerous_images = random.sample(os.listdir(non_cancerous_dir), 5)

def show_images(image_list, label, title, folder):
    """Görüntüleri ekranda gruplar halinde gösterir."""
    plt.figure(figsize=(15, 5))
    for i, img_name in enumerate(image_list):
        img_path = os.path.join(folder, img_name)
        img = mpimg.imread(img_path)
        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(f"Etiket: {label}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Görüntüleri göster
show_images(cancerous_images, "Kanserli", "Kanserli Olarak Seçilen Görüntüler", cancerous_dir)
show_images(non_cancerous_images, "Kansersiz", "Kansersiz Olarak Seçilen Görüntüler", non_cancerous_dir)
