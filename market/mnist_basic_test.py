import copy
import keras
import time

##############################
# Setting random seed for reproducibility
# Seed value
# Apparently you may use different seed values at each stage
seed_value = 100

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
##############################

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
		  epochs = 1,
		  shuffle = False)

t0 = time.time()
# Deep copy whole model object
#model_copy = copy.deepcopy(model)

# Save model and reload
#model.save('temp.keras')

# Copy model and optimizer weights
model_weights = model.get_weights()
optimizer_weights = model.optimizer.get_weights()

model.fit(train_data, train_labels,
		  epochs = 5,
		  shuffle = False)

print(model.evaluate(test_data, test_labels))

# Deep copy whole model object
#model = model_copy

# Save model and reload
#model = keras.models.load_model('temp.keras')

# Copy model and optimizer weights
model.set_weights(model_weights)
model.optimizer.set_weights(optimizer_weights)

model.fit(train_data, train_labels,
		  epochs = 5,
		  shuffle = False)

print(model.evaluate(test_data, test_labels))

t1 = time.time()
print(t1 - t0)
