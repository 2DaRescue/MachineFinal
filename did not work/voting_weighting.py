import os
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Path to your trained model files
model_dir = './batchs/'  # Directory where your models are stored
model_paths = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.keras')]

# Create results directory if it doesn't exist
results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

# Path to your image dataset directory
images = './Data_set/Images'

# Define ImageDataGenerator for validation data
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
    validation_split=0.2  # Use the same validation split as before
)

# Create validation data generator
val_gen = datagen.flow_from_directory(
    directory=images,
    target_size=(128, 128),    # Resize all images to 128x128
    batch_size=32,             # Use same batch size
    class_mode='categorical',  # Multi-class classification
    subset='validation',       # Use the validation subset
    shuffle=False              # Ensure consistent order for metrics
)

# True labels
true_labels = val_gen.classes

# Iterate through each model and predict on the validation data
for model_path in model_paths:
    print(f"Evaluating model: {model_path}")
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Predict on the validation data
    preds = model.predict(val_gen, verbose=1)
    
    # Convert probabilities to class labels
    pred_labels = np.argmax(preds, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    
    # Print metrics
    print(f"Metrics for {os.path.basename(model_path)}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Save metrics to a file
    metrics_path = os.path.join(results_dir, f"{os.path.basename(model_path).replace('.keras', '_metrics.txt')}")
    with open(metrics_path, 'w') as f:
        f.write(f"Metrics for {os.path.basename(model_path)}:\n")
        f.write(f"  Accuracy: {accuracy:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall: {recall:.4f}\n")
        f.write(f"  F1 Score: {f1:.4f}\n")
    
    # Save predictions to a file
    output_path = os.path.join(results_dir, f"{os.path.basename(model_path).replace('.keras', '_preds.txt')}")
    with open(output_path, 'w') as f:
        for pred in preds:
            f.write(','.join(map(str, pred)) + '\n')
    
    print(f"Predictions saved to {output_path}")
    print(f"Metrics saved to {metrics_path}")

print("\nEvaluation completed for all models.")
