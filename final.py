import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the dataset
crop_data_df = pd.read_csv('Crop_recommendation_1.csv').dropna()
X = crop_data_df.drop(columns=['label']).values
y = pd.get_dummies(crop_data_df['label']).values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
tf.keras.utils.plot_model(
    model,
    to_file="model.png",
)
# Save the model
model.save("crop_recommendation_model.h5")

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.show()

plot_training_history(history)

# Plot confusion matrix and classification report
def plot_confusion_matrix_and_report(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    class_names = ['Wheat', 'Rice', 'Maize', 'Barley', 'Millet', 'Sorghum', 'Pigeon Peas', 'Chickpeas', 'Lentil', 'Black gram', 'Green gram', 'Pomegranate', 'Banana', 'Mango', 'Grapes', 'Watermelon', 'Muskmelon', 'Apple', 'Papaya', 'Coconut', 'Orange', 'Peach']

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Capture classification report as text
    report_text = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=class_names)

    # Display classification report as text within the plot or figure
    plt.figure(figsize=(10, 6))
    plt.text(0.1, 0.5, report_text, {'fontsize': 12}, fontfamily='monospace')
    plt.axis('off')  # Turn off axes to display only text
    plt.title('Classification Report')
    plt.show()

# Load and preprocess the dataset (assuming X_test and y_test are defined)
# model = ...  # Load your trained model

# Call the function with the trained model and test data
plot_confusion_matrix_and_report(model, X_test, y_test)