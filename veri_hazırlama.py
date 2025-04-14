
import os
import shutil
import random
base_dir = "C:\\Users\\fidan\\OneDrive\\Masaüstü\\colon_image_sets"
cancerous_dir = os.path.join(base_dir, "colon_aca")
non_cancerous_dir = os.path.join(base_dir, "colon_n")

train_dir = "data/train"
test_dir = "data/test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

train_cancerous_dir = os.path.join(train_dir, "cancerous")
train_non_cancerous_dir = os.path.join(train_dir, "non_cancerous")
test_cancerous_dir = os.path.join(test_dir, "cancerous")
test_non_cancerous_dir = os.path.join(test_dir, "non_cancerous")

os.makedirs(train_cancerous_dir, exist_ok=True)
os.makedirs(train_non_cancerous_dir, exist_ok=True)
os.makedirs(test_cancerous_dir, exist_ok=True)
os.makedirs(test_non_cancerous_dir, exist_ok=True)

cancerous_images = os.listdir(cancerous_dir)
non_cancerous_images = os.listdir(non_cancerous_dir)

train_cancerous = random.sample(cancerous_images, int(0.7 * len(cancerous_images)))
train_non_cancerous = random.sample(non_cancerous_images, int(0.7 * len(non_cancerous_images)))

test_cancerous = [img for img in cancerous_images if img not in train_cancerous]
test_non_cancerous = [img for img in non_cancerous_images if img not in train_non_cancerous]

for img in train_cancerous:
    shutil.copy(os.path.join(cancerous_dir, img), train_cancerous_dir)
for img in train_non_cancerous:
    shutil.copy(os.path.join(non_cancerous_dir, img), train_non_cancerous_dir)

for img in test_cancerous:
    shutil.copy(os.path.join(cancerous_dir, img), test_cancerous_dir)
for img in test_non_cancerous:
    shutil.copy(os.path.join(non_cancerous_dir, img), test_non_cancerous_dir)

print(" Train ve Test veri klasörleri oluşturuldu!")
