import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

data = []
labels = []

IMAGE_SIZE = 64

cat_path = "dataset/cats"
dog_path = "dataset/dogs"

for img in os.listdir(cat_path):
    path = os.path.join(cat_path, img)

    image = cv2.imread(path)

    if image is not None:
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image.flatten()

        data.append(image)
        labels.append(0)

for img in os.listdir(dog_path):
    path = os.path.join(dog_path, img)

    image = cv2.imread(path)

    if image is not None:
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image.flatten()

        data.append(image)
        labels.append(1)

X = np.array(data)
y = np.array(labels)

print("Dataset Loaded Successfully")
print("Total Images:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = SVC(kernel="linear")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred)

with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.colorbar()

plt.savefig("outputs/confusion_matrix.png")

plt.close()

joblib.dump(model, "models/svm_model.pkl")

print("\nModel Saved Successfully")

sample = X_test[0].reshape(1, -1)

prediction = model.predict(sample)

if prediction[0] == 0:
    print("\nPrediction: CAT")
else:
    print("\nPrediction: DOG")

print("\nProject Completed Successfully")