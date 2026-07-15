import numpy as np
import matplotlib.pyplot as plt

# Try importing dependencies; print clear setup help if missing
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    import emnist
except ImportError:
    print("\n[!] Missing dependencies. Please run: pip install tensorflow emnist matplotlib numpy")
    exit(1)

# ==========================================
# STEP 1: LOAD & PREPROCESS EMNIST DATA
# ==========================================
print("Fetching EMNIST Letters dataset... (This auto-downloads on first run)")
# Extracting letters subset: 26 classes (representing A to Z)
x_train, y_train = emnist.extract_training_samples('letters')
x_test, y_test = emnist.extract_test_samples('letters')

# Convert labels from 1-based index (1-26) to 0-based index (0-25) for categorical cross-entropy
y_train = y_train - 1
y_test = y_test - 1

# Transpose and rotate the images 
# EMNIST IDX files are rotated/flipped relative to normal vision orientations due to initial raw data arrays.
print("Applying standard image orientation processing...")
x_train = np.array([np.rot90(np.fliplr(img)) for img in x_train])
x_test = np.array([np.rot90(np.fliplr(img)) for img in x_test])

# Scale pixel values down to [0.0, 1.0] for model stability
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape to include the channel dimension (height, width, channels)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

num_classes = 26
print(f"Train dataset shape: {x_train.shape}")
print(f"Test dataset shape: {x_test.shape}")

# ==========================================
# STEP 2: DEFINE THE CNN ARCHITECTURE
# ==========================================
def build_cnn(input_shape=(28, 28, 1), num_classes=26):
    model = models.Sequential([
        # Block 1: Conv -> Normalization -> MaxPool
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flattening & Dense (Classification) Layer
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Softmax classification head
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_cnn(input_shape=(28, 28, 1), num_classes=num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==========================================
# STEP 3: MODEL TRAINING
# ==========================================
# We train for 5 epochs as a quick benchmark; increase to 15+ for production accuracy
epochs = 5
batch_size = 128

print(f"\nTraining model for {epochs} epochs...")
history = model.fit(
    x_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(x_test, y_test)
)

# ==========================================
# STEP 4: MODEL ACCURACY ASSESSMENT
# ==========================================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("\n" + "="*40)
print("             MODEL EVALUATION            ")
print("="*40)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss:     {test_loss:.4f}")
print("="*40)

# ==========================================
# STEP 5: VISUALIZE MODEL PREDICTIONS
# ==========================================
# Generate predictions
predictions = model.predict(x_test[:6])
predicted_labels = np.argmax(predictions, axis=1)

# Convert integer indices back to actual alphabets (A-Z)
def get_char(label_index):
    return chr(label_index + 65) # ASCII code 65 represents 'A'

plt.figure(figsize=(10, 5))

# Plot training curves
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy Score')
plt.title('Training & Validation Accuracy')
plt.legend()

# Plot sample test predictions
plt.subplot(1, 2, 2)
for i in range(6):
    plt.subplot(2, 6, i + 7)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    true_char = get_char(y_test[i])
    pred_char = get_char(predicted_labels[i])
    plt.title(f"True: {true_char}\nPred: {pred_char}", fontsize=9, color='green' if true_char == pred_char else 'red')
    plt.axis('off')

plt.tight_layout()
plt.show()