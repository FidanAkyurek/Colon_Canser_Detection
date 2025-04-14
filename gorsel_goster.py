import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_images(image_list, labels, title):
    """Görüntüleri ekranda gruplar halinde gösterir."""
    plt.figure(figsize=(15, 5))
    for i, (img_name, label) in enumerate(zip(image_list, labels)):
        img_path = img_name
        img = mpimg.imread(img_path)
        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(f"Etiket: {label}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Görüntülerin bulunduğu dizinler
ana_dizin ="C:\\Users\\fidan\\OneDrive\\Masaüstü\\colon_image_sets"

kanserli_dizin = os.path.join(ana_dizin, "colon_aca")
kansersiz_dizin = os.path.join(ana_dizin, "colon_n")

# Rastgele 5'er kanserli ve kansersiz görüntü seç
kanserli_gorseller = random.sample(os.listdir(kanserli_dizin), 5)
kansersiz_gorseller = random.sample(os.listdir(kansersiz_dizin), 5)

# Dosya yollarını oluştur
kanserli_yollar = [os.path.join(kanserli_dizin, img) for img in kanserli_gorseller]
kansersiz_yollar = [os.path.join(kansersiz_dizin, img) for img in kansersiz_gorseller]

# Kanserli görüntüleri göster
show_images(kanserli_yollar, ["1"] * 5, "Kanserli Görüntüler")

# Kansersiz görüntüleri göster
show_images(kansersiz_yollar, ["0"] * 5, "Kansersiz Görüntüler")
