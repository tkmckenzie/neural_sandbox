import keras

num_epochs = 1

(train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()

train_data = train_data / 255
test_data = test_data / 255

train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

model = keras.Sequential([
		keras.layers.Flatten(),
		
		keras.layers.Dense(32),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
#		keras.layers.Dropout(0.5),
		
		keras.layers.Dense(64),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
#		keras.layers.Dropout(0.5),
		
		keras.layers.Dense(10, activation = 'softmax')
])

model.compile(keras.optimizers.Adam(),
			  loss = keras.losses.categorical_crossentropy,
			  metrics = ['accuracy'])

model.fit(train_data, train_labels,
		  epochs = num_epochs)

print(model.evaluate(test_data, test_labels))
