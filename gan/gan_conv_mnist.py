import keras
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Fitting parameters
num_epochs = 5
noise_dim = 100

lr_discriminator = 0.0001
beta1_discriminator = 0.5
lr_generator = 0.0001
beta1_generator = 0.5

# Load data
(train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()

# Normalize image data
train_data = train_data / 255

# Reshape train data for batches
batch_size = 32
num_batches = int(train_data.shape[0] / batch_size)
train_data = train_data[:(batch_size * num_batches),:,:]

train_data = train_data.reshape((num_batches, batch_size) + train_data.shape[1:])

# Reshape to include channels
train_data = train_data.reshape(train_data.shape + tuple([1]))
image_shape = train_data.shape[2:]

# Generator model
model_generator = keras.Sequential([
	keras.layers.Dense(64 * 7 * 7, input_shape = (noise_dim,)),
	keras.layers.Reshape((7, 7, 64)),
	
	keras.layers.Conv2DTranspose(32, kernel_size = 3, strides = 2, padding = 'same'),
	keras.layers.BatchNormalization(),
	keras.layers.LeakyReLU(0.01),
	
	keras.layers.Conv2DTranspose(16, kernel_size = 3, strides = 1, padding = 'same'),
	keras.layers.BatchNormalization(),
	keras.layers.LeakyReLU(0.01),
	
	keras.layers.Conv2DTranspose(1, kernel_size = 3, strides = 2, padding = 'same'),
	
	keras.layers.Activation('tanh')
])
model_generator.compile(keras.optimizers.Adam(lr = lr_generator, beta_1 = beta1_generator), keras.losses.binary_crossentropy)

# DEBUG
n_gen_trainable = len(model_generator.trainable_weights)

# Discriminator model
model_discriminator = keras.Sequential([
	keras.layers.Conv2D(32, kernel_size = 3, strides = 2, padding = 'same', input_shape = image_shape),
	keras.layers.BatchNormalization(),
	keras.layers.LeakyReLU(0.01),
	
	keras.layers.Conv2D(64, kernel_size = 3, strides = 1, padding = 'same'),
	keras.layers.BatchNormalization(),
	keras.layers.LeakyReLU(0.01),
	
	keras.layers.Flatten(),
	keras.layers.Dense(32),
	keras.layers.ReLU(),
	
	keras.layers.Dense(16),
	keras.layers.ReLU(),
	
	keras.layers.Dense(1, activation = 'sigmoid')
])
model_discriminator.compile(keras.optimizers.Adam(lr = lr_discriminator, beta_1 = beta1_discriminator), keras.losses.binary_crossentropy, metrics = ['accuracy'])

# DEBUG
n_disc_trainable = len(model_discriminator.trainable_weights)

# Stacked model
#model_discriminator_fixed = keras.models.Model(inputs = model_discriminator.inputs, outputs = model_discriminator.outputs)
model_discriminator_fixed = keras.engine.network.Network(inputs = model_discriminator.inputs, outputs = model_discriminator.outputs)
model_discriminator_fixed.trainable = False
model = keras.Sequential([model_generator, model_discriminator_fixed])
model.compile(keras.optimizers.Adam(lr = lr_generator, beta_1 = beta1_generator), keras.losses.binary_crossentropy, metrics = ['accuracy'])

# DEBUG
n_disc_fixed_trainable = len(model_discriminator_fixed.trainable_weights)
n_model_trainable = len(model.trainable_weights)

assert(n_model_trainable == n_gen_trainable)
assert(n_disc_fixed_trainable == 0)

#epoch = 1
#asdf
generator_mean_acc = 1
discriminator_mean_acc = 1
batch_order = np.arange(num_batches, dtype = int)
for epoch in range(num_epochs):
	loss_discriminator = np.zeros((num_batches, 2))
	loss_generator = np.zeros((num_batches, 2))
	np.random.shuffle(batch_order)
	for batch_index in tqdm(range(num_batches), desc = 'Epoch %i' % (epoch + 1)):
		batch = batch_order[batch_index]
		
		true_samples_batch = train_data[batch,:,:,:,:]
		gen_noise = np.random.normal(size = (batch_size, noise_dim))
		gen_samples = model_generator.predict(gen_noise)
		
		x_combined = np.concatenate((true_samples_batch, gen_samples))
		y_combined = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))))
		
		loss_discriminator[batch,:] = model_discriminator.train_on_batch(x_combined, y_combined)
		
		noise = np.random.normal(size = (batch_size * 2, noise_dim))
		y_mislabeled = np.ones((batch_size * 2, 1))
		
		loss_generator[batch,:] = model.train_on_batch(noise, y_mislabeled)
    
	generator_mean_acc = np.mean(loss_generator[:,1])
	discriminator_mean_acc = np.mean(loss_discriminator[:,1])
	print ('[Discriminator :: accuracy: %f], [ Generator :: accuracy: %f]' % (discriminator_mean_acc, generator_mean_acc))
	
	if (epoch + 1) % 1 == 0:
		gen_noise = np.random.normal(size = (16, noise_dim))
		gen_samples = model_generator.predict(gen_noise)
		
		
		plt.figure(figsize = (10, 10))
		for i in range(16):
			plt.subplot(4, 4, i + 1)
			plt.imshow(gen_samples[i,:,:,0])
		plt.savefig('images/gan_%i.png' % (epoch + 1))
		plt.close()

# Validation
