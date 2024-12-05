import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from joblib import load
import os

# Load the saved model
model_path = 'mobilenet_dog_breeds.joblib'  # Path to your saved model
model = load(model_path)
print(f"Model loaded successfully from {model_path}")

# Prepare the data generator for the existing dataset
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2  # Assume we split the existing data into train/validation
)

# Directory containing dataset
dataset_dir = './Data_set/Images'  # Update with the correct path

# Create test data generator
test_gen = datagen.flow_from_directory(
    directory=dataset_dir,
    target_size=(224, 224),  # Match model input size
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Disable shuffle to maintain consistent results
)

# Evaluate the model on the test dataset
def evaluate_model():
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

# Generate a classification report
def generate_classification_report():
    # Predict on all test images
    predictions = model.predict(test_gen)
    predicted_classes = np.argmax(predictions, axis=1)  # Get class indices
    true_classes = test_gen.classes  # True class labels
    class_labels = list(test_gen.class_indices.keys())  # Class names

    # Generate classification report
    print("\nClassification Report:")
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

# Generate a confusion matrix
def plot_confusion_matrix_simplified():
    # Predict on all test images
    predictions = model.predict(test_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes

    # Generate confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
    plt.title("Simplified Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()

# Visualize predictions on a batch
def visualize_predictions():
    batch_images, batch_labels = next(test_gen)
    batch_predictions = model.predict(batch_images)
    predicted_classes = np.argmax(batch_predictions, axis=1)

    for i in range(5):  # Visualize first 5 images in the batch
        plt.imshow(batch_images[i])
        true_label = list(test_gen.class_indices.keys())[np.argmax(batch_labels[i])]
        predicted_label = list(test_gen.class_indices.keys())[predicted_classes[i]]
        plt.title(f"True: {true_label} | Pred: {predicted_label}")
        plt.axis('off')
        plt.show()

# Plot training and validation metrics
def plot_metrics(history_path='training_history.npy'):
    # Load history file if available
    if os.path.exists(history_path):
        history = np.load(history_path, allow_pickle=True).item()
        # Plot loss
        plt.figure(figsize=(10, 5))
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title("Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # Plot accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title("Accuracy Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
    else:
        print("Training history file not found.")

if __name__ == '__main__':
    print("Evaluating the model...")
    evaluate_model()

    print("\nGenerating classification report...")
    generate_classification_report()

    print("\nPlotting confusion matrix...")
    plot_confusion_matrix_simplified()

    print("\nVisualizing predictions...")
    visualize_predictions()

    print("\nPlotting training metrics...")
    plot_metrics()
