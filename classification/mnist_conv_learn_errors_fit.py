import keras
import numpy as np
import pickle as pkl

batch_size = 50
epochs_init = 10
epochs_error = 100
epochs_cat = 10

(train_data, train_labels_arg), (test_data, test_labels_arg) = keras.datasets.mnist.load_data()
train_labels = keras.utils.to_categorical(train_labels_arg)
test_labels = keras.utils.to_categorical(test_labels_arg)

train_data = train_data / 255
test_data = test_data / 255

train_data = np.reshape(train_data, train_data.shape + tuple([1]))
test_data = np.reshape(test_data, test_data.shape + tuple([1]))

input_shape = train_data.shape[1:]

num_classes = train_labels.shape[1]

# Specify models
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
model_error = keras.Sequential([
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
model_cat = keras.Sequential([
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
        keras.layers.Dense(1, activation = 'sigmoid')
])


# Compile model
model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.adam(),
              metrics = ['accuracy'])
model_error.compile(loss = keras.losses.categorical_crossentropy,
					optimizer = keras.optimizers.adam(),
					metrics = ['accuracy'])
model_cat.compile(loss = keras.losses.binary_crossentropy,
				  optimizer = keras.optimizers.adam(),
				  metrics = ['accuracy'])

# Fit initial model
model.fit(train_data, train_labels,   
          batch_size = batch_size,
          epochs = epochs_init,
          verbose = 1)

# Find errors
predictions = model.predict(train_data)
predictions_arg = np.apply_along_axis(np.argmax, 1, predictions)
error_obs = np.where(predictions_arg != train_labels_arg)[0]

train_data_error = train_data[error_obs,:,:,:]
train_labels_error = train_labels[error_obs,:]

# Fit error data
model_error.fit(train_data_error, train_labels_error,
				batch_size = batch_size,
				epochs = epochs_error,
				verbose = 1)

# Determine which model fits each observation best
predictions_error = model_error.predict(train_data)

loss = list(map(lambda i: model.evaluate(train_data[None,i,:,:,:], train_labels[None,i,:], verbose = 0)[0], range(train_data.shape[0])))
loss_error = list(map(lambda i: model_error.evaluate(train_data[None,i,:,:,:], train_labels[None,i,:], verbose = 0)[0], range(train_data.shape[0])))

cat_labels = np.apply_along_axis(np.argmax, 0, np.array([loss, loss_error])) # Indicator of whether original model fits better than error model

# Fit categorization model
model_cat.fit(train_data, cat_labels,
			  batch_size = batch_size,
			  epochs = epochs_cat,
			  verbose = 1)

# Validate
predictions_cat = model_cat.predict(test_data)

obs_original = np.where(predictions_cat > 0.5)[0]
obs_error = np.setdiff1d(range(test_data.shape[0]), obs_original)

test_data_original = test_data[obs_original,:,:,:]
test_labels_original = test_labels[obs_original,:]
test_data_error = test_data[obs_error,:,:,:]
test_labels_error = test_labels[obs_error,:]

acc_original = model.evaluate(test_data_original, test_labels_original)[1]
acc_error = model_error.evaluate(test_data_error, test_labels_error)[1]

prop_original = test_data_original.shape[0] / test_data.shape[0]
prop_error = test_data_error.shape[0] / test_data.shape[0]

print(prop_original * acc_original + prop_error * acc_error)
