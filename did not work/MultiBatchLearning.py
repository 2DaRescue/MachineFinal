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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

######################                      ######################
######################   Getting the Data   ######################
######################                      ######################

# Paths to images and annotations
images = './Data_set/Images'
annotation = './Data_set/Annotation'

# Count files in the images and annotations folders
print(f"{GREEN} The Data \n Files in images folder: {count_files(images)}")
print(f"Files in annotation folder: {count_files(annotation)}")

# Pair images and annotations
data = pair_files(images, annotation)

# Split data into batches of classifications (e.g., 10 classes per batch)
# Assuming folders are named after class labels
class_names = sorted(os.listdir(images))  # List of class labels
batch_size = 10  # Number of classes per batch
class_batches = [class_names[i:i + batch_size] for i in range(0, len(class_names), batch_size)]

print(f"\nTotal Classes: {len(class_names)}")
print(f"Batch Size: {batch_size}")
print(f"Number of Batches: {len(class_batches)}")

# Display a random image to check
random_pair = random.choice(data)
print(f"Random Pair: {random_pair} \n")
show_image(random_pair['image'], random_pair['annotation'])

# Print the names of the folders in each batch
#for idx, batch in enumerate(class_batches):
#    print(f"{GREEN}Batch {idx + 1}:{RESET}")
#    for class_name in batch:
#        print(f"  {class_name}")
#    print()  # Add a blank line between batches

######################                              ######################
######################   Splitting Data by Batches  ######################
######################                              ######################

# Temporary datagen for batch splitting
basic_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0  # Normalize pixel values to [0, 1]
)

for batch_idx, batch_classes in enumerate(class_batches):
    print(f"\n{BLUE}Processing Batch {batch_idx + 1} with classes: {batch_classes}{RESET}")

    # Filter data for current batch classes
    batch_data = [
        {'image': item['image'], 'class_name': item['annotation']['class_name']}
        for item in data
        if any(class_name in item['image'] for class_name in batch_classes)
    ]

    print(f"Total images in Batch {batch_idx + 1}: {len(batch_data)}")

    # Convert batch_data to a DataFrame
    batch_df = pd.DataFrame(batch_data)

    # Split into training and validation
    train_df, val_df = train_test_split(batch_df, test_size=0.2, random_state=42, shuffle=True)

    print(f"Training size: {len(train_df)}, Validation size: {len(val_df)}")

    # Prepare data generators for this batch
    train_gen = basic_datagen.flow_from_dataframe(
        train_df,  # Use train data for generator
        x_col='image',
        y_col='class_name',  # Extracted class name
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    val_gen = basic_datagen.flow_from_dataframe(
        val_df,  # Use validation data for generator
        x_col='image',
        y_col='class_name',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    # Output summary
    print(f"Training batches: {len(train_gen)}")
    print(f"Validation batches: {len(val_gen)}")
######################                          ######################
######################   Incremental Training   ######################
######################                          ######################

# Initialize a base model for the first batch
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze base layers for feature extraction

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(class_batches[0]), activation='softmax')(x)  # Output layer for the first batch
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train and update the model for each batch
for batch_idx, batch_classes in enumerate(class_batches):
    print(f"\n{GREEN}Training on Batch {batch_idx + 1}/{len(class_batches)} with classes: {batch_classes}{RESET}")
    
    # Filter and preprocess the current batch
    batch_data = [
        {'image': item['image'], 'class_name': item['annotation']['class_name']}
        for item in data
        if any(class_name in item['image'] for class_name in batch_classes)
    ]
    batch_df = pd.DataFrame(batch_data)
    train_df, val_df = train_test_split(batch_df, test_size=0.2, random_state=42, shuffle=True)

    train_gen = basic_datagen.flow_from_dataframe(
        train_df, x_col='image', y_col='class_name', target_size=(128, 128), batch_size=32, class_mode='categorical'
    )
    val_gen = basic_datagen.flow_from_dataframe(
        val_df, x_col='image', y_col='class_name', target_size=(128, 128), batch_size=32, class_mode='categorical'
    )

    # Update output layer for the new batch
    if batch_idx > 0:
        output = Dense(len(batch_classes), activation='softmax')(model.layers[-2].output)
        model = Model(inputs=model.input, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model on the current batch
    history = model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1)

    # Save the updated model
    model.save(f'model_batch_{batch_idx + 1}.keras')
    print(f"{BLUE}Model for Batch {batch_idx + 1} saved successfully!{RESET}")

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(val_gen)
    print(f"{BLUE}Batch {batch_idx + 1} - Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}{RESET}")
    
######################                                          ######################
######################      Final Evaluation on All Classes     ######################
######################                                          ######################

# Consolidate validation data from all batches
all_val_data = []
for batch_idx, batch_classes in enumerate(class_batches):
    batch_data = [
        {'image': item['image'], 'class_name': item['annotation']['class_name']}
        for item in data
        if any(class_name in item['image'] for class_name in batch_classes)
    ]
    _, val_data = train_test_split(batch_data, test_size=0.2, random_state=42, shuffle=True)
    all_val_data.extend(val_data)

# Convert to DataFrame for generator
all_val_df = pd.DataFrame(all_val_data)

# Create a validation data generator for all classes
all_val_gen = basic_datagen.flow_from_dataframe(
    all_val_df,
    x_col='image',
    y_col='class_name',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Evaluate the model
print(f"\n{GREEN}Evaluating on All Classes{RESET}")
final_loss, final_accuracy = model.evaluate(all_val_gen)
print(f"Final Validation Loss: {final_loss}")
print(f"Final Validation Accuracy: {final_accuracy}")

######################                                          ######################
######################      Fine-Tuning the Entire Model        ######################
######################                                          ######################

# Unfreeze the base model for fine-tuning
base_model.trainable = True

# Compile the model with a smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune the model
print(f"\n{GREEN}Fine-Tuning the Model{RESET}")
fine_tune_history = model.fit(
    train_gen,  # Use the training generator with all classes
    validation_data=all_val_gen,  # Use the consolidated validation generator
    epochs=3,  
    verbose=1
)

# Save the fine-tuned model
model.save('final_fine_tuned_model.keras')
print(f"{GREEN}Fine-tuned model saved successfully!{RESET}")
