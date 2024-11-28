import random
from funct import pair_files, show_image, count_files, preprocess_images
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Input

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
# Step 4: Model Building and Training
######################

# Define the model architecture
model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Regularization
    Dense(train_gen.num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Use 'binary_crossentropy' for 2 classes
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,              # Number of epochs to train
    verbose=1               # Progress output during training
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_gen)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Save the model (optional)
model.save('./model/dog_breed_classifier.keras')