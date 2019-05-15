import keras
import numpy as np
import scipy.stats as sps


N = 100 # Number of samples to draw at once
batch_size = 32 # Number of batches in each epoch
num_batches = 500
num_epochs = 50

# Generate training data
def sampler(size): return 5 * np.random.normal(size = size) - 2
true_samples = sampler((num_batches, batch_size, N))

# Generator model
model_generator = keras.Sequential([
	keras.layers.Dense(256, input_shape = (N,)),
	keras.layers.LeakyReLU(alpha = 0.2),
	keras.layers.BatchNormalization(momentum = 0.8),
	
	keras.layers.Dense(512),
	keras.layers.LeakyReLU(alpha = 0.2),
	keras.layers.BatchNormalization(momentum = 0.8),
	
	keras.layers.Dense(1024),
	keras.layers.LeakyReLU(alpha = 0.2),
	keras.layers.BatchNormalization(momentum = 0.8),
	
	keras.layers.Dense(N)
])
model_generator.compile(keras.optimizers.Adam(), keras.losses.mean_squared_error)

# Discriminator model
model_discriminator = keras.Sequential([
	keras.layers.Dense(N, input_shape = (N,)),
	keras.layers.LeakyReLU(alpha = 0.2),
	keras.layers.Dense(int(N / 2)),
	keras.layers.LeakyReLU(alpha = 0.2),
	keras.layers.Dense(1, activation = 'sigmoid')
])
model_discriminator.compile(keras.optimizers.Adam(), keras.losses.binary_crossentropy, metrics = ['accuracy'])

# Stacked model
model_discriminator.trainable = False
model = keras.Sequential([model_generator, model_discriminator])
model.compile(keras.optimizers.Adam(), keras.losses.binary_crossentropy)

#epoch = 1
#asdf
batch_order = np.arange(num_batches, dtype = int)
for epoch in range(num_epochs):
	loss_discriminator = np.zeros((num_batches, 2))
	loss_generator = np.zeros(num_batches)
	np.random.shuffle(batch_order)
	for batch_index in range(num_batches):
		batch = batch_order[batch_index]
		
		true_samples_batch = true_samples[batch,:,:]
		gen_noise = np.random.normal(size = (batch_size, N))
		gen_samples = model_generator.predict(gen_noise)
		
		x_combined = np.concatenate((true_samples_batch, gen_samples))
		y_combined = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))))
		
		loss_discriminator[batch,:] = model_discriminator.train_on_batch(x_combined, y_combined)
		
		noise = np.random.normal(size = (batch_size, N))
		y_mislabeled = np.ones((batch_size, 1))
		
		loss_generator[batch] = model.train_on_batch(noise, y_mislabeled)
		
	print ('epoch: %d, [Discriminator :: accuracy: %f], [ Generator :: loss: %f]' % (epoch + 1, np.mean(loss_discriminator[:,1]), np.mean(loss_generator)))

# Validation using a KS test
num_samples_validation = 10

true_samples_validation = sampler((num_samples_validation, N))

gen_noise = np.random.normal(size = (num_samples_validation, N))
gen_samples_validation = model_generator.predict(gen_noise)

true_samples_validation = true_samples_validation.flatten()
gen_samples_validation = gen_samples_validation.flatten()

print(sps.ks_2samp(true_samples_validation, gen_samples_validation))
