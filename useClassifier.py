
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from joblib import load
import os
import random
from sklearn.preprocessing import label_binarize

# no nonsenes warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Load 
model_path = 'mobilenet_dog_breeds_Tuned.joblib'  # Path to the saved model
model = load(model_path)
print("Model loaded")

# Prepare the data generator for preprocessing and validation data
#nomilize the photo and also more splitting for validation. 
# small augments
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  
    validation_split=0.2  
)

# Directory containing the dataset
dataset = './Data_set/Images'  
metrics_dir = './METRICS'
# Creates a data generator for the validation subset of the dataset. 
# resizes images to 224x224 s
# processes them in batches of 32 images.
# cool stufff

test_gen = datagen.flow_from_directory(
    directory=dataset,
    target_size=(224, 224),  # Resize images to match model input size
    batch_size=32,  # Process data in batches of 32
    class_mode='categorical',  # Use categorical labels for multi-class classification
    subset='validation',  # Use validation split of the data
    shuffle=False  # Keep order consistent for reproducible results
)

# Evaluate the model
print("\nEvaluating the model...")
test_loss, test_accuracy = model.evaluate(test_gen)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Generate predictions and create a classification report
# Predict labels for all validation data true vs predicted stuff.

print("\nGenerating classification report...")
predictions = model.predict(test_gen) 
predicted_classes = np.argmax(predictions, axis=1)  
true_classes = test_gen.classes  
class_labels = list(test_gen.class_indices.keys())  

# Print classification report
print("\nClassification Report:")
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

with open(os.path.join(metrics_dir, "classification_report.txt"), 'w') as f:
    f.write(report)

# Generate and plot the confusion matrix
# rather ungl
print("\nPlotting confusion matrix...")
cm = confusion_matrix(true_classes, predicted_classes) 
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)  
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

# Visualize random predictions from the test dataset
print("\n 10 random predictions...")
filenames = test_gen.filenames  # List of filenames in the test dataset
true_labels = test_gen.classes  # True labels corresponding to the filenames

# Select random images to visualize predictions
# Choose 10 random images
# Add batch dimension for prediction
random_indices = random.sample(range(len(filenames)), 10)  
for idx in random_indices:
    img_path = os.path.join(test_gen.directory, filenames[idx])  
    img = load_img(img_path, target_size=(224, 224))  
    img_array = img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

# Predicted class name and True class name
    prediction = model.predict(img_array)  
    predicted_class_idx = np.argmax(prediction)  
    predicted_label = class_labels[predicted_class_idx]  
    true_label = class_labels[true_labels[idx]]  

    # show pups
    plt.imshow(img)
    plt.title(f"True: {true_label} | Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()
    


# Precision-Recall Curve
print("\nPrecision-Recall Curve")
y_true_bin = label_binarize(true_classes, classes=range(len(class_labels)))
y_pred_bin = label_binarize(predicted_classes, classes=range(len(class_labels)))

# Plot Precision-Recall curve for each class
for i in range(len(class_labels)):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
    plt.plot(recall, precision, label=f'Class {class_labels[i]}')

plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig(os.path.join(metrics_dir, 'precision_recall_curve.png'))
plt.show()

# ROC Curve and AUC
print("\n ROC Curve")
fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
plt.savefig(os.path.join(metrics_dir, 'roc_curve.png'))


