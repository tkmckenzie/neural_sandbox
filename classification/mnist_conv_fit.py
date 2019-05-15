import keras
import numpy as np
import pickle as pkl

batch_size = 32
epochs = 5

(train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

train_data = train_data / 255
test_data = test_data / 255

train_data = np.reshape(train_data, train_data.shape + tuple([1]))
test_data = np.reshape(test_data, test_data.shape + tuple([1]))

input_shape = train_data.shape[1:]

num_classes = train_labels.shape[1]

# Specify model
model = keras.Sequential([
		keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = input_shape),
		keras.layers.MaxPooling2D(pool_size = (2, 2)),
		keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
		keras.layers.MaxPooling2D(pool_size = (2, 2)),
		keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
		keras.layers.MaxPooling2D(pool_size = (2, 2)),
		keras.layers.Flatten(),
		keras.layers.Dense(64, activation = 'relu'),
		keras.layers.Dropout(0.5),
		keras.layers.Dense(128, activation = 'relu'),
		keras.layers.Dropout(0.5),
		keras.layers.Dense(num_classes, activation = 'sigmoid')
])

# Compile model
model.compile(loss = keras.losses.categorical_crossentropy,
			  optimizer = keras.optimizers.adam(),
			  metrics = ['accuracy'])

# Fit model
model.fit(train_data, train_labels,
		  validation_data = (test_data, test_labels),   
		  batch_size = batch_size,
		  epochs = epochs,
		  verbose = 1)
