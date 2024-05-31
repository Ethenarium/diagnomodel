import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import time

df = pd.read_csv('cad.csv')

num_classes = 21
X = df.drop(['Diagnosis Result'], axis=1)
y = keras.utils.to_categorical(df['Diagnosis Result'], num_classes)

model = Sequential()
model.add(layers.InputLayer(input_shape=(X.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

start_time = time.time()
history = model.fit(X, y, epochs=50, batch_size=32, verbose=2)
end_time = time.time()

print("Time taken for training:", end_time - start_time, "seconds")

model.save('pad_model.keras')
