import tensorflow as tf
from keras import Sequential, applications, layers, callbacks
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

img_shape = 224
batch_size = 128
train_data_path = "data/train"

train_preprocessor = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255.,
    rotation_range=10,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_data = train_preprocessor.flow_from_directory(
    train_data_path,
    class_mode="categorical",
    target_size=(img_shape, img_shape),
    color_mode='rgb',
    shuffle=True,
    batch_size=batch_size,
    subset='training',
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

checkpoint_path = "model.keras"
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

model.save("Final.keras")

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