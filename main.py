import random
from funct import pair_files, show_image, count_files
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler

from joblib import dump

from sklearn.model_selection import StratifiedShuffleSplit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

######################                      ######################
######################   Getting the Data   ######################
######################                      ######################

#images and annotation (square around the dog)
images = './Data_set/Images'
annotation = './Data_set/Annotation'
 
#did not end up using because we split our own data randomly. just a note.. 
file_list_path = './Data_set/list/file_list'
train_list_path = './Data_set/list/train_list'
test_list_path = './Data_set/list/test_list'

# Count files in the images and annotations folders
print(f"{GREEN} The Data \n Files in images folder: {count_files(images)}")
print(f"Files in annotation folder: {count_files(annotation)}")

# pairing the image and the annotation so we can see the square around the dog :)
data = pair_files(images, annotation)

# Display a random image to check 
random_pair = random.choice(data)
print(f"Random Pair: {random_pair} \n")
show_image(random_pair['image'], random_pair['annotation'])


######################                          ######################
######################   Splitting the Data     ###################### 
######################                          ######################


# Create stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(data, [os.path.basename(d['image']).split('_')[0] for d in data]):
    train_data = [data[i] for i in train_idx]
    test_data = [data[i] for i in test_idx]

# Output summary
print(f"{BLUE}\n Spliting the data \n Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

# Inspect a few samples - Training Data
print(f"\nTraining Data Sample:")
for item in train_data[:3]:  # Loop to print each item on a new line
    print(item)

# Inspect a few samples - Test Data
print(f"\nTest Data Sample:")
for item in test_data[:3]:  # Loop to print each item on a new line
    print(item)

# Print the lengths of the splits again
print(f"\nTraining set size: {len(train_data)}")
print(f"Test set size: {len(test_data)} \n")

######################                                          ######################
######################      Step 3: Preprocessing the Data      ######################
######################                                          ######################

# Initialize ImageDataGenerator with normalization
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,        # Rotate images up to 30 degrees
    width_shift_range=0.2,    # Shift images horizontally
    height_shift_range=0.2,   # Shift images vertically
    shear_range=0.2,          # Apply shearing
    zoom_range=0.2,           # Zoom in/out
    brightness_range=(0.8, 1.2),  # Adjust brightness
    horizontal_flip=True,     # Flip images horizontally
    validation_split=0.2      # 20% validation split
)

train_gen = datagen.flow_from_directory(
    directory=images,
    target_size=(224, 224),  # Increased resolution
    batch_size=64,  # Increased batch size
    class_mode='categorical',
    subset='training'
)

# Prepare validation data generator
val_gen = datagen.flow_from_directory(
    directory=images,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'        # Specify this is the validation split
)

# Output summary of data generators
print(f"{GREEN}Preping Data \n Training batches: {len(train_gen)}")
print(f"Validation batches: {len(val_gen)}")


######################
# Step 4: Transfer Learning
######################
## mobile net is a CNN 
# Load the MobileNet model without the top layer
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

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



def scheduler(epoch, lr):
    return float(lr * tf.math.exp(-0.1))  # Ensures output is a float
    


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,  # Stop training if no improvement after 3 epochs
    restore_best_weights=True
)

lr_scheduler = LearningRateScheduler(scheduler)

# Add it to the model training step
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[lr_scheduler,early_stopping],  # Add scheduler here
    verbose=1
)


# Save the model using joblib
dump(model, 'mobilenet_dog_breeds.joblib')
print("Model saved successfully as 'mobilenet_dog_breeds.joblib'")

print("Model saved successfully!")


# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_gen)
print(f"{BLUE}Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}{RESET}")

######################
# Step 5: Visualizations
######################
# Visualize training and validation metrics
import matplotlib.pyplot as plt

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

plt.figure()
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(accuracy, label="Training Accuracy")
plt.plot(val_accuracy, label="Validation Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()