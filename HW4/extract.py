import tensorflow as tf

model = tf.keras.models.load_model('model.keras')
model.save_weights('model.weights.h5')