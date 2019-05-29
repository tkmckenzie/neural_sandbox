import keras
import market_classes
import numpy as np

# Market and fitting parameters
num_firms = 5
minutes_per_firm = 2

num_periods = 10

# Load and clean data
(train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()

train_data = train_data / 255
test_data = test_data / 255

train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

# Set up consumer object
consumer = market_classes.Consumer(test_data, test_labels, minutes_per_firm * num_firms * 60, train_data.shape[0], lambda eval_result: 1 / (1 - eval_result['accuracy']**2))

# Set up firm objects
num_hidden_layers = [2] * num_firms
neurons_per_layer = [np.random.randint(1, 100, num_hidden_layers[i]) for i in range(num_firms)]
activations = [[keras.layers.ReLU()] * num_hidden_layers[i] for i in range(num_firms)]
dropout = [[False] * num_hidden_layers[i] for i in range(num_firms)]
normalization = [[True] * num_hidden_layers[i] for i in range(num_firms)]

firms = [market_classes.Firm(minutes_per_firm * 60, train_data, train_labels) for i in range(num_firms)]
for i in range(num_firms):
	firms[i].initialize_model(num_hidden_layers[i], neurons_per_layer[i], activations[i], dropout[i], normalization[i])

# Perform initial evaluation
for firm in firms:
	firm.set_evaluate_result(firm.evaluate(consumer.x_val, consumer.y_val))

## Iterate for single firm
#asdf
#i = 0
#firms[i].training_decision(consumer, market_classes.exclude_element(firms, i))

for period in range(num_periods):
	print('\nPeriod', period + 1)
	# Update firms' models
	decisions = [True] * num_firms
	num_updates = 0
	while any(decisions):
		decisions = [firms[i].training_decision(consumer, market_classes.exclude_element(firms, i)) for i in range(num_firms)]
		print(decisions)
		num_updates += sum(decisions)
	
	# Produce output
	consumer.budget = consumer.base_budget
	for i in range(num_firms):
		firms[i].produce(consumer, market_classes.exclude_element(firms, i))
	
	
	print('Accuracies:\n', [np.round(firm.get_evaluate_result()['accuracy'], 3) for firm in firms])
	print('Budgets:\n', [np.round(firm.budget, 1) for firm in firms])
	print('Updates:', num_updates)
