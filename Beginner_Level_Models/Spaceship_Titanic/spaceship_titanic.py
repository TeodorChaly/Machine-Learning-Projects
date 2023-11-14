import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

training_dataset_df = pd.read_csv('path/train.csv')
test_dataset_df = pd.read_csv('path/test.csv')
print("Full train dataset shape is {}".format(training_dataset_df.shape))

training_dataset_df.head()

labels = training_dataset_df.get(["Transported"])

features = ["CryoSleep",  "VIP"]

X = pd.get_dummies(training_dataset_df[features])
X_test = pd.get_dummies(test_dataset_df[features])


for i in X["CryoSleep_False"]:
  if i != 0 and i != 1:
    print(i)
# train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X, label=labels)
# test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_test, label=labels)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, labels, epochs=25, batch_size=32, validation_split=0.2)

predictions = model.predict(X_test)

predictions = (predictions > 0.5).astype(int).flatten()

output = pd.DataFrame({'PassengerId': test_dataset_df.PassengerId, 'Transported': predictions})
output.to_csv('/content/drive/MyDrive/Code/Datasets/spaceship-titanic_data_set/output.csv', index=False)
print("Your submission was successfully saved!")