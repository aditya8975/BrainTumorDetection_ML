import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")

# Function to load training data
def train_df(tr_path):
    classes, class_paths = zip(*[(label, os.path.join(tr_path, label, image))
                                 for label in os.listdir(tr_path) if os.path.isdir(os.path.join(tr_path, label))
                                 for image in os.listdir(os.path.join(tr_path, label))])
    tr_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return tr_df

# Function to load testing data
def test_df(ts_path):
    classes, class_paths = zip(*[(label, os.path.join(ts_path, label, image))
                                 for label in os.listdir(ts_path) if os.path.isdir(os.path.join(ts_path, label))
                                 for image in os.listdir(os.path.join(ts_path, label))])
    ts_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return ts_df

# Load the data
tr_df = train_df('E:/MCA/proj/BTumor/Training')
ts_df = test_df('E:/MCA/proj/BTumor/Testing')

# Visualize the count of images in each class
plt.figure(figsize=(15,7))
ax = sns.countplot(data=tr_df , y=tr_df['Class'])
ax.bar_label(ax.containers[0])
plt.title('Count of images in each class (Training)', fontsize=20)
plt.show()

plt.figure(figsize=(15, 7))
ax = sns.countplot(y=ts_df['Class'], palette='viridis')
ax.bar_label(ax.containers[0])
plt.title('Count of images in each class (Testing)', fontsize=20)
plt.show()

# Split the data
valid_df, ts_df = train_test_split(ts_df, train_size=0.5, random_state=20, stratify=ts_df['Class'])

# Image generators
batch_size = 32
img_size = (299, 299)

_gen = ImageDataGenerator(rescale=1/255, brightness_range=(0.8, 1.2))
ts_gen = ImageDataGenerator(rescale=1/255)

tr_gen = _gen.flow_from_dataframe(tr_df, x_col='Class Path', y_col='Class', batch_size=batch_size, target_size=img_size)
valid_gen = _gen.flow_from_dataframe(valid_df, x_col='Class Path', y_col='Class', batch_size=batch_size, target_size=img_size)
ts_gen = ts_gen.flow_from_dataframe(ts_df, x_col='Class Path', y_col='Class', batch_size=16, target_size=img_size, shuffle=False)

# Class mapping
class_dict = tr_gen.class_indices
classes = list(class_dict.keys())
images, labels = next(ts_gen)

# Visualize images
plt.figure(figsize=(20, 20))
for i, (image, label) in enumerate(zip(images, labels)):
    plt.subplot(4, 4, i + 1)
    plt.imshow(image)
    class_name = classes[np.argmax(label)]
    plt.title(class_name, color='k', fontsize=15)
plt.show()

# Define the model
img_shape = (299, 299, 3)

base_model = tf.keras.applications.Xception(
    include_top=False, 
    weights='E:/MCA/proj/BTumor/xception_weights_tf_dim_ordering_tf_kernels_notop.h5', 
    input_shape=img_shape, 
    pooling='max'
)


model = Sequential([
    base_model,
    Flatten(),
    Dropout(rate=0.3),
    Dense(128, activation='relu'),
    Dropout(rate=0.25),
    Dense(4, activation='softmax')
])

model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

model.summary()

# Train the model
hist = model.fit(tr_gen, epochs=10, validation_data=valid_gen, shuffle=False)

# Save the model
output_dir = 'E:/MCA/proj/BTumor/brain_tumor_model_tf'
model.save(output_dir, save_format='tf')

# Compress the model
import shutil
shutil.make_archive('E:/MCA/proj/BTumor/brain_tumor_model_tf', 'zip', 'E:/MCA/proj/BTumor/brain_tumor_model_tf')

# Plot training history
tr_acc = hist.history['accuracy']
tr_loss = hist.history['loss']
tr_per = hist.history['precision']
tr_recall = hist.history['recall']
val_acc = hist.history['val_accuracy']
val_loss = hist.history['val_loss']
val_per = hist.history['val_precision']
val_recall = hist.history['val_recall']

# Plot performance
plt.figure(figsize=(20, 12))
plt.style.use('fivethirtyeight')

Epochs = [i + 1 for i in range(len(tr_acc))]
loss_label = f'Best epoch = {str(np.argmin(val_loss) + 1)}'
acc_label = f'Best epoch = {str(np.argmax(val_acc) + 1)}'
per_label = f'Best epoch = {str(np.argmax(val_per) + 1)}'
recall_label = f'Best epoch = {str(np.argmax(val_recall) + 1)}'

# Plot Loss
plt.subplot(2, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label='Training loss')
plt.plot(Epochs, val_loss, 'g', label='Validation loss')
plt.scatter(np.argmin(val_loss) + 1, np.min(val_loss), s=150, c='blue', label=loss_label)
plt.title('Training and Validation Loss')
plt.legend()

# Plot Accuracy
plt.subplot(2, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
plt.scatter(np.argmax(val_acc) + 1, np.max(val_acc), s=150, c='blue', label=acc_label)
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot Precision
plt.subplot(2, 2, 3)
plt.plot(Epochs, tr_per, 'r', label='Precision')
plt.plot(Epochs, val_per, 'g', label='Validation Precision')
plt.scatter(np.argmax(val_per) + 1, np.max(val_per), s=150, c='blue', label=per_label)
plt.title('Precision and Validation Precision')
plt.legend()

# Plot Recall
plt.subplot(2, 2, 4)
plt.plot(Epochs, tr_recall, 'r', label='Recall')
plt.plot(Epochs, val_recall, 'g', label='Validation Recall')
plt.scatter(np.argmax(val_recall) + 1, np.max(val_recall), s=150, c='blue', label=recall_label)
plt.title('Recall and Validation Recall')
plt.legend()

plt.suptitle('Model Training Metrics Over Epochs', fontsize=16)
plt.show()

# Evaluate the model
train_score = model.evaluate(tr_gen, verbose=1)
valid_score = model.evaluate(valid_gen, verbose=1)
test_score = model.evaluate(ts_gen, verbose=1)

print(f"Train Loss: {train_score[0]:.4f}")
print(f"Train Accuracy: {train_score[1]*100:.2f}%")
print(f"Validation Loss: {valid_score[0]:.4f}")
print(f"Validation Accuracy: {valid_score[1]*100:.2f}%")
print(f"Test Loss: {test_score[0]:.4f}")
print(f"Test Accuracy: {test_score[1]*100:.2f}%")

# Predict and show confusion matrix
preds = model.predict(ts_gen)
y_pred = np.argmax(preds, axis=1)
cm = confusion_matrix(ts_gen.classes, y_pred)
labels = list(class_dict.keys())
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('Truth Label')
plt.show()

# Classification report
clr = classification_report(ts_gen.classes, y_pred)
print(clr)

# Prediction function
def predict(img_path):
    label = list(class_dict.keys())
    plt.figure(figsize=(12, 12))
    img = Image.open(img_path)
    resized_img = img.resize((299, 299))
    img = np.asarray(resized_img)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    predictions = model.predict(img)
    probs = list(predictions[0])
    labels = label
    plt.subplot(2, 1, 1)
    plt.imshow(resized_img)
    plt.subplot(2, 1, 2)
    bars = plt.barh(labels, probs)
    plt.xlabel('Probability', fontsize=15)
    ax = plt.gca()
    ax.bar_label(bars, fmt='%.2f')
    plt.show()

# Example predictions


predict('E:\MCA\proj\BTumor\Testing\meningioma\Te-meTr_0000.jpg')
predict('E:\MCA\proj\BTumor\Testing\glioma\Te-glTr_0007.jpg')
predict('E:\MCA\proj\BTumor\Testing\notumor\Te-noTr_0001.jpg')
predict('E:\MCA\proj\BTumor\Testing\pituitary\Te-piTr_0001.jpg')
