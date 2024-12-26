import tensorflow as tf
import numpy as np
import os
from sklearn.utils import resample
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential, applications, layers, callbacks
import warnings

warnings.filterwarnings("ignore")

train_data_path = "data/train"
balanced_data_path = "data/train_balanced"
img_shape = 224
batch_size = 128

train_preprocessor = ImageDataGenerator(
    rescale=1 / 255.,
    rotation_range=10,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
)

train_data = train_preprocessor.flow_from_directory(
    train_data_path,
    class_mode="categorical",
    target_size=(img_shape, img_shape),
    color_mode='rgb',
    shuffle=False,
    batch_size=1
)

class_counts = {k: 0 for k in train_data.class_indices}
for _, labels in train_data:
    class_idx = np.argmax(labels)
    class_name = list(train_data.class_indices.keys())[class_idx]
    class_counts[class_name] += 1
    if train_data.batch_index == 0:
        break

print("Original class distribution:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

max_count = max(class_counts.values())

Path(balanced_data_path).mkdir(parents=True, exist_ok=True)

for class_name in class_counts.keys():
    class_dir = Path(train_data_path) / class_name
    class_files = list(class_dir.glob("*"))
    
    oversampled_files = resample(class_files, n_samples=max_count, replace=True, random_state=42)

    balanced_class_dir = Path(balanced_data_path) / class_name
    balanced_class_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, file_path in enumerate(oversampled_files):
        file_copy_path = balanced_class_dir / f"{class_name}_{idx}.jpg"
        if not file_copy_path.exists():
            tf.io.gfile.copy(str(file_path), str(file_copy_path))

print("Dataset balanced and saved to:", balanced_data_path)

class_counts = Counter({
    class_name: len(os.listdir(os.path.join(balanced_data_path, class_name)))
    for class_name in os.listdir(balanced_data_path)
    if os.path.isdir(os.path.join(balanced_data_path, class_name))
})

print("Balanced class distribution:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

is_balanced = len(set(class_counts.values())) == 1
if is_balanced:
    print("\nThe dataset is perfectly balanced!")
else:
    print("\nThe dataset is not balanced. Please review the script or class folders.")

train_data_path = "data/train_balanced"
batch_size = 128

train_preprocessor = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255.,
    rotation_range=10,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2,
)

train_data = train_preprocessor.flow_from_directory(
    train_data_path,
    class_mode="categorical",
    target_size=(img_shape, img_shape),
    color_mode='rgb',
    shuffle=True,
    batch_size=batch_size,
    subset='training'
)

validation_data = train_preprocessor.flow_from_directory(
    train_data_path,
    class_mode="categorical",
    target_size=(img_shape, img_shape),
    color_mode='rgb',
    shuffle=False,
    batch_size=batch_size,
    subset='validation'
)

base = applications.ResNet50V2(input_shape=(img_shape, img_shape, 3), include_top=False, weights='imagenet')
base.trainable = True

for layer in base.layers[:-70]:
    layer.trainable = False

def Create_Model():
    model = Sequential([
        base,
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax'),
    ])
    return model

model = Create_Model()
model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_path = "model_balanced.keras"
Checkpoint = callbacks.ModelCheckpoint(
    checkpoint_path, monitor="val_accuracy", save_best_only=True
)

Early_Stopping = callbacks.EarlyStopping(
    monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1
)

Reducing_LR = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=2, verbose=1
)

callback = [Checkpoint, Early_Stopping, Reducing_LR]

steps_per_epoch = len(train_data)
validation_steps = len(validation_data)

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=50,
    callbacks=callback,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

model.save("Final_balanced.keras")

acc = np.array(history.history['accuracy'])
val_acc = np.array(history.history['val_accuracy'])
loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

plt.savefig("Unbalanced.png")