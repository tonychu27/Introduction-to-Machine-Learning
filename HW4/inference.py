import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from keras import applications, layers, Sequential

test_data_path = "data/test"
img_shape = 224

test_image_files = [os.path.join(test_data_path, fname) for fname in os.listdir(test_data_path) if fname.endswith(".jpg")]

base = applications.ResNet50V2(input_shape=(img_shape, img_shape, 3), include_top=False, weights='imagenet')

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

model.load_weights('model_balanced.weights.h5')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

def preprocess_image(image_path, target_size=(img_shape, img_shape, 3)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

predictions = []
file_names = []

for image_path in tqdm(test_image_files):
    preprocessed_image = preprocess_image(image_path)

    prediction = model.predict(preprocessed_image, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = predicted_class

    file_name = os.path.basename(image_path).replace(".jpg", "")
    
    file_names.append(file_name)
    predictions.append(predicted_label)

df = pd.DataFrame({
    "filename": file_names,
    "label": predictions
})

df.to_csv("prediction.csv", index=False)
print("Predictions saved to prediction.csv")