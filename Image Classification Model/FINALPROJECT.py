# ==========================================
# SETUP (Replace these with your actual objects/paths)
# ==========================================
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Example placeholders - replace these with your actual data
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_dir = 'path_to_your_test_data'
# train_generator = ... 
# extract_feat_model = ...
# fine_tune_model = ...
# extract_feat_history = ...
# fine_tune_history = ...

# ==========================================
# Task 1: Print the version of TensorFlow
# ==========================================
print(f"Task 1 - TensorFlow Version: {tf.__version__}")


# ==========================================
# Task 2: Create a `test_generator` using the `test_datagen` object
# ==========================================
# Replace 'test_dir' with your actual test directory path
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)


# ==========================================
# Task 3: Print the length of the `train_generator`
# ==========================================
print(f"Task 3 - Length of train_generator: {len(train_generator)}")


# ==========================================
# Task 4: Print the summary of the model
# ==========================================
# Usually refers to the active model you are working with
model.summary()


# ==========================================
# Task 5: Compile the model
# ==========================================
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    metrics=['acc']
)
print("Task 5 - Model compiled successfully.")


# ==========================================
# Task 6: Plot accuracy curves for training and validation sets (extract_feat_model)
# ==========================================
acc = extract_feat_history.history['acc']
val_acc = extract_feat_history.history['val_acc']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Task 6 - Accuracy: Extract Features Model')
plt.legend()
plt.show()


# ==========================================
# Task 7: Plot loss curves for training and validation sets (fine tune model)
# ==========================================
loss = fine_tune_history.history['loss']
val_loss = fine_tune_history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Task 7 - Loss: Fine-Tune Model')
plt.legend()
plt.show()


# ==========================================
# Task 8: Plot accuracy curves for training and validation sets (fine tune model)
# ==========================================
acc_ft = fine_tune_history.history['acc']
val_acc_ft = fine_tune_history.history['val_acc']
epochs = range(1, len(acc_ft) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, acc_ft, 'bo', label='Training acc')
plt.plot(epochs, val_acc_ft, 'b', label='Validation acc')
plt.title('Task 8 - Accuracy: Fine-Tune Model')
plt.legend()
plt.show()


# ==========================================
# Task 9: Plot a test image using Extract Features Model (index_to_plot = 1)
# ==========================================
index_to_plot = 1
images, labels = next(test_generator)
img = images[index_to_plot]

# Make prediction
pred_score = extract_feat_model.predict(np.expand_dims(img, axis=0))[0][0]
pred_label = "Class 1" if pred_score > 0.5 else "Class 0"

plt.imshow(img)
plt.title(f"Task 9 - Extract Feat Model Prediction: {pred_label}")
plt.axis('off')
plt.show()


# ==========================================
# Task 10: Plot a test image using Fine-Tuned Model (index_to_plot = 1)
# ==========================================
# Re-using index 1 and the same image batch
img_ft = images[index_to_plot]

# Make prediction
pred_score_ft = fine_tune_model.predict(np.expand_dims(img_ft, axis=0))[0][0]
pred_label_ft = "Class 1" if pred_score_ft > 0.5 else "Class 0"

plt.imshow(img_ft)
plt.title(f"Task 10 - Fine-Tuned Model Prediction: {pred_label_ft}")
plt.axis('off')
plt.show()