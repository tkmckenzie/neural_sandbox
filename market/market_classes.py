import keras
import time

class BankruptError(Exception):
	pass

class Firm:
	# Firms need:
	#    1) Model
	#    2) Budget to spend fitting model
	#    3) Budget to spend evaluating model
	#    4) Method to learn from other firms
	#    5) Method to choose fitting another epoch vs evaluating
	def __init__(self, budget, x_train, y_train):
		# budget: (real) Time available to spend, in seconds
		self.budget = budget
		self.x_train = x_train
		self.y_train = y_train
		self.model_initialized = False
	
	def initialize_model(self, num_hidden_layers, neurons_per_layer, activations, dropout, normalization):
		# num_hidden_layers: (int) Number of hidden layers
		# neurons_per_layer: (list of ints, len = num_hidden_layers) Number of neurons per hidden layer
		# activations: (list of keras activations, len = num_hidden_layers) Activation functions to use in each layer
		# dropout: (list of bools, len = num_hidden_layers) Indicator of whether dropout should be used
		# normalization: (list of bools, len = num_hidden_layers) Indicator of whether regularization should be used
		
		# Store values so learner can access later
		self.num_hidden_layers = num_hidden_layers
		self.neurons_per_layer = neurons_per_layer
		self.activations = activations
		self.dropout = dropout
		self.normalization = normalization
		
		# Set up and build model
		self.model = keras.Sequential()
		self.model.add(keras.layers.Flatten())
		
		for layer in range(num_hidden_layers):
			self.model.add(keras.layers.Dense(neurons_per_layer[layer]))
			if normalization[layer]: self.model.add(keras.layers.BatchNormalization())
			self.model.add(activations[layer])
			if dropout[layer]: self.model.add(keras.layers.Dropout(0.5))
		
		self.model.add(keras.layers.Dense(10, activation = 'softmax'))
		
		# Compile model
		self.model.compile(optimizer = keras.optimizers.Adam(),
					 loss = keras.losses.categorical_crossentropy,
					 metrics = ['accuracy'])
		
		# Other objects to change
		self.model_initialized = True
	
	def fit(self):
		# Updates model by one epoch and returns time it took to fit
		
		if self.budget < 0: raise BankruptError
		
		# Fit one epoch and time
		t0 = time.time()
		self.model.fit(self.x_train, self.y_train, verbose = 0)
		t1 = time.time()
		
		return t1 - t0
	
	def evaluate(self, x, y):
		t0 = time.time()
		evaluate_result = self.model.evaluate(x, y, verbose = 0)
		t1 = time.time()
		
		return {'avg_time': (t1 - t0) / x.shape[0], 'accuracy': evaluate_result[1]}
	
	def produce(self, x, y, units_demanded):
		# units_demanded: (positive real) Number of units demanded by consumer
		
		if self.budget < 0: raise BankruptError
		
		# Evaluate accuracy and time it takes to evaluate
		evaluate_result = self.evaluate(x, y)
		
		# Calculate total costs
		cost = evaluate_result['avg_time'] * units_demanded
		
		# Return accuracy
		return evaluate_result['accuracy']
	
	def training_decision(self, consumer, other_firms):
		# consumer: (Consumer object) Consumer that demands output
		# other_firms: (list of Firm objects) Other firms in competition
		
		model_weights_init = self.model.get_weights()
		optimizer_weights_init = self.model.optimizer.get_weights()
		
		evaluate_result_init = self.evaluate(consumer.x_val, consumer.y_val)
		
		if evaluate_result_init['avg_time'] * consumer.units_demanded <= self.budget:
			training_cost = self.fit()
			evaluate_result_post = self.evaluate(consumer.x_val, consumer.y_val)
			
			if training_cost + evaluate_result_post['avg_time'] * consumer.units_demanded <= self.budget:
				other_firm_accuracy = [firm.evaluate(consumer.x_val, consumer.y_val) for firm in other_firms]
				demand_init = consumer.demand_i(evaluate_result_init['accuracy'], other_firm_accuracy)
				demand_post = consumer.demand_i(evaluate_result_post['accuracy'], other_firm_accuracy)
				
				if demand_post > demand_init:
					
		
class Consumer:
	def __init__(self, x_val, y_val, budget, units_demanded, utility_function):
		# budget: (positive real) Time available to spend per period
		self.x_val = x_val
		self.y_val = y_val
		self.budget = budget
		self.units_demanded = units_demanded
		self.utility_function = utility_function
	
	def demand_i(self, firm_accuracy, other_firm_accuracy):
		NotImplementedError