import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import json
import os
# ======================
# CONFIG
# ======================
img_size = 64
batch_size = 32

base_path = r"C:\Users\NITHYAA SRI\OneDrive\Documents\STS\STS\dataset"

train_path = os.path.join(base_path, "Traindata")
val_path = os.path.join(base_path, "testdata")

# ======================
# CHECK PATHS (IMPORTANT)
# ======================
print("Train exists:", os.path.exists(train_path))
print("Val exists:", os.path.exists(val_path))

# ======================
# DATA GENERATORS
# ======================
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# ======================
# MODEL
# ======================
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

# ======================
# COMPILE
# ======================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ======================
# TRAIN
# ======================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# ======================
# SAVE MODEL
# ======================
model.save("sign_model.h5")

# ======================
# SAVE LABELS
# ======================
with open("labels.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("Training completed successfully ✅")