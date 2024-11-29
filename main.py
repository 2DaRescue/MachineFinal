import random
from funct import pair_files, show_image, count_files
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Input

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D



######################
# Step 1: Getting the Data
######################

# Paths to images, annotations, and lists
images = './Data_set/Images'
annotation = './Data_set/Annotation'
file_list_path = './Data_set/list/file_list'
train_list_path = './Data_set/list/train_list'
test_list_path = './Data_set/list/test_list'

# Count files in the images and annotations folders
print(f"Files in images folder: {count_files(images)}")
print(f"Files in annotation folder: {count_files(annotation)}")

# Pair the files (basic pairing)
data = pair_files(images, annotation)

# Display a random image and its annotation
random_pair = random.choice(data)
print(f"Random Pair: {random_pair}")
show_image(random_pair['image'], random_pair['annotation'])

######################
# Step 2: Splitting the Data
######################

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

# Output summary
print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

# Inspect a few samples
print("\nTraining Data Sample:")
print(train_data[:3])

print("\nTest Data Sample:")
print(test_data[:3] )

# Print the lengths of the splits
print(f"\nTraining set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

######################
# Step 3: Preprocessing the Data
######################
# Initialize ImageDataGenerator with normalization
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
    validation_split=0.2  # Split 80% training, 20% validation
)

# Prepare training data generator
train_gen = datagen.flow_from_directory(
    directory=images,
    target_size=(128, 128),    # Resize all images to 128x128
    batch_size=32,             # Process images in batches of 32
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    subset='training'          # Specify this is the training split
)

# Prepare validation data generator
val_gen = datagen.flow_from_directory(
    directory=images,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'        # Specify this is the validation split
)

# Output summary of data generators
print(f"Training batches: {len(train_gen)}")
print(f"Validation batches: {len(val_gen)}")

######################
# Step 4: Transfer Learning
######################

# Load the MobileNet model without the top layer
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the base model layers to use them as feature extractors
base_model.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global pooling to reduce feature maps
x = Dense(128, activation='relu')(x)  # Fully connected layer
output = Dense(train_gen.num_classes, activation='softmax')(x)  # Output layer

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,  # Adjust epochs as needed
    verbose=1
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_gen)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")