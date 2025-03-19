import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import random

# Read the JSON file
with open('internal_medicine_diseases.json', 'r') as file:
    data = json.load(file)

# Convert the diseases list to a DataFrame
df = pd.json_normalize(data['diseases'])

# Create dummy variables for symptoms using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
symptoms_encoded = pd.DataFrame(
    mlb.fit_transform(df['symptoms']),
    columns=mlb.classes_,
    index=df.index
)

# Function to augment data
def augment_data(original_df, symptoms_encoded, num_samples=50000):
    augmented_data = []
    disease_names = original_df['name'].tolist()
    all_symptoms = list(symptoms_encoded.columns)
    
    for _ in range(num_samples):
        # Randomly select a disease
        disease = random.choice(disease_names)
        original_symptoms = original_df[original_df['name'] == disease]['symptoms'].iloc[0]
        
        # Get the core symptoms for this disease (70% chance of keeping each core symptom)
        core_symptoms = [symptom for symptom in original_symptoms if random.random() < 0.7]
        
        # Add some random non-core symptoms (20% chance for each disease to have 1-3 additional symptoms)
        if random.random() < 0.2:
            num_additional = random.randint(1, 3)
            additional_symptoms = random.sample([s for s in all_symptoms if s not in original_symptoms], num_additional)
            core_symptoms.extend(additional_symptoms)
        
        augmented_data.append({
            'name': disease,
            'symptoms': core_symptoms
        })
    
    return pd.DataFrame(augmented_data)

# Augment the data
augmented_df = augment_data(df, symptoms_encoded)


# Create dummy variables for augmented data
mlb = MultiLabelBinarizer()
symptoms_encoded_aug = pd.DataFrame(
    mlb.fit_transform(augmented_df['symptoms']),
    columns=mlb.classes_,
    index=augmented_df.index
)
symptoms_encoded_aug.to_csv('augmented_diseases.csv', index=False)

# Encode disease names
le = LabelEncoder()
disease_encoded = le.fit_transform(augmented_df['name'])

# Prepare features (X) and target (y)
X = symptoms_encoded_aug.values
y = tf.keras.utils.to_categorical(disease_encoded)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model with reduced complexity
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(y.shape[1], activation='softmax')
])

# Compile the model with a lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')

# Function to predict disease from symptoms
def predict_disease(symptoms_list):
    # Convert symptoms to binary format
    input_symptoms = np.zeros((1, len(mlb.classes_)))
    for symptom in symptoms_list:
        if symptom in mlb.classes_:
            idx = list(mlb.classes_).index(symptom)
            input_symptoms[0, idx] = 1
    
    # Get prediction
    prediction = model.predict(input_symptoms)
    top_3_idx = prediction[0].argsort()[-3:][::-1]
    
    print("\nTop 3 predicted diseases:")
    for idx in top_3_idx:
        disease_name = le.inverse_transform([idx])[0]
        confidence = prediction[0][idx] * 100
        print(f"{disease_name}: {confidence:.2f}%")

# Test with 5 random diseases
print("\nTesting with 5 random diseases:")
test_diseases = random.sample(df['name'].tolist(), 5)
for disease in test_diseases:
    original_symptoms = df[df['name'] == disease]['symptoms'].iloc[0]
    # Randomly select 3-5 symptoms from the disease
    num_symptoms = random.randint(3, 5)
    test_symptoms = random.sample(list(original_symptoms), min(num_symptoms, len(original_symptoms)))
    print(f"\nTesting disease: {disease}")
    print(f"Selected symptoms: {test_symptoms}")
    predict_disease(test_symptoms)

# Save the model
model.save('disease_prediction_model')

# Print model summary
print("\nModel Architecture:")
model.summary()

# Save the augmented dataset
augmented_df.to_csv('augmented_diseases.csv', index=False)

# Calculate sparsity of the dataset
sparsity = (symptoms_encoded_aug.values == 0).sum() / (symptoms_encoded_aug.shape[0] * symptoms_encoded_aug.shape[1])

# Save the ML-ready dataset
ml_ready_df = pd.concat([augmented_df['name'], symptoms_encoded_aug], axis=1)
ml_ready_df.to_csv('diseases_ml_ready.csv', index=False)

# Create a correlation matrix between symptoms
corr_matrix = symptoms_encoded_aug.corr()
# Get upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Find top 10 highest correlations
highest_corr = upper.unstack().sort_values(ascending=False)[:10] 