import keras
import market_classes

(train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()

train_data = train_data / 255
test_data = test_data / 255

train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)


firm = market_classes.Firm(100)
firm.initialize_model(num_hidden_layers = 2,
					  neurons_per_layer = [32, 64],
					  activations = [keras.layers.ReLU(), keras.layers.ReLU()],
					  dropout = [False, False],
					  normalization = [True, True])

firm.fit(train_data, train_labels)
firm.fit(train_data, train_labels)

print(firm.training_times)
print(firm.budget)

print(firm.evaluate(test_data, test_labels))
