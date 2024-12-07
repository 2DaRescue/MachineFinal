from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


model = load_model('mobilenet_dog_breeds.keras')  # Make sure you load the correct model
print("MobileNet model loaded successfully!")


dataset = './Data_set/Images'  # Dataset directory


# Unfreeze only the last 10 layers for fine-tuning
for layer in model.layers[-10:]:
    layer.trainable = True

# Add custom layers
x = model.output
x = Dense(128, activation='relu', name='dense_finetune_1')(x)  # Custom dense layer
x = Dropout(0.5, name='dropout_finetune_1')(x)  # Dropout layer
output = Dense(120, activation='softmax', name='output_finetune')(x)  # Final output layer

# Recreate the model with the added layers
model = Model(inputs=model.input, outputs=output)

# Compile the model with a higher learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-3),  # Higher learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#got this rom main. to continue learning
# after adjusting.
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  
)

train_gen = datagen.flow_from_directory(
    directory=dataset,
    target_size=(224, 224),  
    batch_size=64,
    class_mode='categorical',
    subset='training'  
)

val_gen = datagen.flow_from_directory(
    directory=dataset,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    subset='validation'  
)

# early stop and reduced learning if needed.
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6)

# Fine-tune the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Save the fine-tuned model
model.save('mobilenet_dog_breeds_Tuned.joblib')
print("saved")

# re-Evaluate 
val_loss, val_accuracy = model.evaluate(val_gen)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")