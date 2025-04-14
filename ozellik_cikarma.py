
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tqdm import tqdm

base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

train_dir = "data/train"
test_dir = "data/test"

X = []
Y = []

def extract_features(img_path, label):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = base_model.predict(img_array)
    X.append(features.flatten())
    Y.append(label)

for category, label in [("cancerous", 1), ("non_cancerous", 0)]:
    category_dir = os.path.join(train_dir, category)
    for img_name in tqdm(os.listdir(category_dir), desc=f"{category} Train Görselleri İşleniyor"):
        extract_features(os.path.join(category_dir, img_name), label)

for category, label in [("cancerous", 1), ("non_cancerous", 0)]:
    category_dir = os.path.join(test_dir, category)
    for img_name in tqdm(os.listdir(category_dir), desc=f"{category} Test Görselleri İşleniyor"):
        extract_features(os.path.join(category_dir, img_name), label)

X = np.array(X)
Y = np.array(Y)
np.save("features.npy", X)
np.save("labels.npy", Y)

print(" Özellik çıkarma tamamlandı ve kaydedildi!")
