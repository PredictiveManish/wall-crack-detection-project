import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import os
from tensorflow.keras.models import Model
base_path = os.path.expanduser("~/Downloads/chirag-project/concrete_data")
train_dir = os.path.join(base_path, "train")
val_dir = os.path.join(base_path, "val")

# Data generators
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

# Base model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
preds = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=preds)

# Freeze base layers for transfer learning
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
model.fit(train_gen, validation_data=val_gen, epochs=5)

# Save model in repo
model_save_path = os.path.expanduser("~/Downloads/crack_detector.h5")
model.save(model_save_path)
print(f"Model saved as {model_save_path}")