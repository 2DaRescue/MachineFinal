import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Paths
results_dir = './results'  # Directory containing model predictions
prediction_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('_preds.txt')]

# Load predictions
all_predictions = []
for file in prediction_files:
    with open(file, 'r') as f:
        preds = [list(map(float, line.strip().split(','))) for line in f]
        all_predictions.append(np.array(preds))

# Stack predictions (shape: [num_models, num_samples, num_classes])
all_predictions = np.stack(all_predictions, axis=0)  # Shape: [12, num_samples, num_classes]

# Average predictions across models for simplicity
aggregated_predictions = np.mean(all_predictions, axis=0)  # Shape: [num_samples, num_classes]

# Ground truth labels (assuming they are the same for all models)
val_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0, validation_split=0.2
).flow_from_directory(
    directory='./Data_set/Images',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

true_labels = val_gen.classes  # True labels

# Prepare data for ensemble model
X = aggregated_predictions
y = to_categorical(true_labels)  # Convert to one-hot encoding

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build ensemble model
ensemble_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

ensemble_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
ensemble_model.summary()

# Train the ensemble model
history = ensemble_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)

# Evaluate on validation data
val_loss, val_accuracy = ensemble_model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Save the ensemble model
ensemble_model.save('./results/ensemble_model.keras')

# Predictions on validation set
y_pred = np.argmax(ensemble_model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)

# Print classification metrics
print("Classification Report:")
print(classification_report(y_true, y_pred))
