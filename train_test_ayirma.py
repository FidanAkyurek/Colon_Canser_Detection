
import numpy as np
from sklearn.model_selection import train_test_split

X = np.load("features.npy")
Y = np.load("labels.npy")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("Y_train.npy", Y_train)
np.save("Y_test.npy", Y_test)

print(" Train-Test veri seti rastgele ayrıldı ve kaydedildi!")
