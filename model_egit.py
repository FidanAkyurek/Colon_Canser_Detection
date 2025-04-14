import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Verileri yükleyelim
X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")
X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")

# MLP Modelini Oluştur
model = keras.Sequential([
    layers.Dense(256, activation="relu", input_shape=(X_train.shape[1],)),  # 1. Gizli Katman
    layers.Dense(128, activation="relu"),  # 2. Gizli Katman
    layers.Dense(1, activation="sigmoid")  # Çıkış Katmanı (Binary Classification)
])

# Modeli Derle
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Modeli Eğit
history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test))
print("\n📈 Eğitim Metrikleri Grafikleri Oluşturuluyor...")

# Doğruluk (Accuracy) Grafiği
plt.figure(figsize=(10, 5)) # Boyut ayarlaması yapıldı
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu (train accuracy)') # Türkçe etiket
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu (validation accuracy)') # Türkçe etiket
plt.title('Model Doğruluk Grafiği (Model Accuracy)') # Başlık güncellendi
plt.ylabel('Doğruluk (Accuracy)')
plt.xlabel('Epoch')
plt.legend(loc='lower right') # Lejantın yerini belirledik
# plt.savefig('mlp_model_accuracy_plot.png') # Kaydetme satırı yorumdan çıkarılabilir
plt.show()

# Kayıp (Loss) Grafiği
plt.figure(figsize=(10, 5)) # Boyut ayarlaması yapıldı
plt.plot(history.history['loss'], label='Eğitim Kaybı (train loss)') # Türkçe etiket
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı (validation loss)') # Türkçe etiket
plt.title('Model Kayıp Grafiği (Model Loss)') # Başlık güncellendi
plt.ylabel('Kayıp (Loss)')
plt.xlabel('Epoch')
plt.legend(loc='upper right') # Lejantın yerini belirledik
# plt.savefig('mlp_model_loss_plot.png') # Kaydetme satırı yorumdan çıkarılabilir
plt.show()

# Test doğruluğu
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"\n📌 Test Doğruluğu: {test_acc:.4f}")

# Modeli Kaydet
model.save("kolondeneme/mlp_model.h5")

print(f"\n📌 Model 'kolondeneme/mlp_model.h5' olarak kaydedildi!")
