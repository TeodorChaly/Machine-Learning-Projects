# Machine-Learning-Projects
Repository for machine learning, deep learning, and neural network projects. My own roadmap in AI area from beginner to advanced.

## Self-Made Models

### 🚀 Next Word Prediction Model

#### 🤔 What it does
The Next Word Prediction Model predicts the next word in a sequence based on the input text, using a pre-trained deep learning model and tokenizer.

#### 📝 Instructions
1. Ensure you have the necessary dependencies installed, including pandas, numpy, keras, and others.
2. Load the pre-trained model and tokenizer using the provided paths.
3. Utilize the `model_predict` function to generate predictions for the next word in a given text sequence.
4. Explore the top predictions with the `model_results` function, specifying the number of predictions to display.

#### 🌐 Examples
```python
# Example usage of the Next Word Prediction Model

# Load the pre-trained model and tokenizer
model, tokenizer = loading_data()

# Generate predictions for the next word in a sequence
predicted = model_predict("The quick brown")

# Display the top 3 predictions
model_results(3)
```


## Beginner level model

### 🚀 Spaceship Titanic Classification Model

#### 🤔 What it does
The Spaceship Titanic Classification Model classifies passengers as "Transported" or not based on features like "CryoSleep" and "VIP."

#### 📝 Instructions
1. Install the necessary dependencies, including tensorflow_decision_forests, tensorflow, pandas, numpy, seaborn, and matplotlib.
2. Load the training and test datasets using the provided paths.
3. Preprocess the data by encoding categorical features and handling missing values.
4. Define and train a simple binary classification model using TensorFlow and Keras.
5. Make predictions on the test dataset and save the results to 'output.csv'.

#### 🌐 Examples
```python
!pip install tensorflow_decision_forests
# Load datasets
train_df = pd.read_csv('/kaggle/input/spaceship-titanic/spaceship-titanic_data_set/train.csv')
test_df = pd.read_csv('/kaggle/input/spaceship-titanic/spaceship-titanic_data_set/test.csv')

# Extract labels and features
labels = train_df['Transported']
features = pd.get_dummies(train_df[['CryoSleep', 'VIP']])

# Train a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(features.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(features, labels, epochs=25, batch_size=32, validation_split=0.2)

# Make predictions on the test dataset
predictions = (model.predict(pd.get_dummies(test_df[['CryoSleep', 'VIP']])) > 0.5).astype(int).flatten()

# Save predictions to 'output.csv'
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Transported': predictions})
output.to_csv('output.csv', index=False)
print("Submission saved!")
```
