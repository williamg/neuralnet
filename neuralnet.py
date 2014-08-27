# Neural Network python module
# Written by William Ganucheau 2014
import sys, random, math

# Utility function to create an empty (zero) array of length n
def zeroArray(n):
	return [0 for i in range(0, n)]

# A neural network class capable of saving/loading from a file,
# Training based on user-provided input-output data,
# and evaluating based on user input
class NeuralNetwork:

	# Create a neural network. If weights aren't provided, they are 
	# initialized to random values
	def __init__(self, neurons, weights=None, biases=None):
		self.numLayers = len(neurons)
		self.neurons = neurons
		self.numNeurons = sum(neurons)
		self.numInput = neurons[0];
		self.numOutput = neurons[len(neurons)-1]
		self.weights = weights
		self.biases = biases
		self.errorGradients = zeroArray(self.numNeurons)
		self.outputs = zeroArray(self.numNeurons)

		self.inputs = []
		for layer in range(0, self.numLayers-1):
			for neuron in range(0, self.neurons[layer+1]):
				self.inputs.append(zeroArray(self.neurons[layer]))

		# Default random values
		if weights == None:
			self.__initWeights()

		self.tempWeights = weights;

		if biases == None:
			self.__initBiases()

	# Initialize random weights for the neural network
	def __initWeights(self):
		i = 0
		self.weights = []

		# Initialize the weights for every non-input neuron
		for layer in range(1, self.numLayers):
			numWeights = self.neurons[layer-1]
			for neuron in range(0, self.neurons[layer]):
				self.weights.append(
					[random.uniform(-0.5, 0.5) for j in range(0, numWeights)]
				)

	# Initialize random biases for the neural network
	def __initBiases(self):
		numBiased = self.numNeurons-self.numInput
		self.biases = [random.uniform(-0.5, 0.5) for j in range(0, numBiased)]

	# Save the neural network to a file
	def save(self, path):
		data = ''

		# First line is # of layers
		data += str(self.numLayers) + '\n'

		# Second line is # of neurons in each layer
		for c in range(0, len(self.neurons)):
			data += str(self.neurons[c]) + ' '
		data += '\n'

		# Third line is biases for all the neurons
		for b in range(0, len(self.biases)):
			data += str(self.biases[b]) + ' '
		data += '\n'

		# Following lines are the weights of each neuron
		i = 0
		for l in range(1,  self.numLayers):
			for n in range(0, self.neurons[l]):
				for w in range (0, len(self.weights[i])):
					data += str(self.weights[i][w]) + ' '
				data += '\n'
				i += 1

		f = open(path, 'w')
		f.write(data)
		f.flush()
		f.close()

	# Load a network from a file
	@classmethod
	def load(self, path):

		f = open(path, 'r')
		numLayers = int(f.readline())
		charNeurons = f.readline().split()
		charBiases = f.readline().split()
		neurons = [int(charNeurons[i]) for i in range(0, len(charNeurons))]
		biases = [float(charBiases[i]) for i in range(0, len(charBiases))]
		weights = zeroArray(sum(neurons))

		for neuron in range(0, sum(neurons)):
			charWeights = f.readline().split()
			weights[neuron] = ( 
				[float(charWeights[i]) for i in range(0, len(charWeights))]
			)

		# Instantiate network
		return self(neurons, weights, biases)

	# Evaluate an input array with the neural network
	def eval(self, input):
		if len(input) != self.numInput:
			sys.exit ('Error: Invalid input size.')

		output = []
		neuronIndex = 0;

		for layer in range(1, self.numLayers):
			output = zeroArray(self.neurons[layer])

			for neuron in range (0, self.neurons[layer]): 
				neuronIndex = self.__getIndex(layer) + neuron
				numWeights = len(self.weights[neuronIndex])

				for weight in range (0, numWeights):
					val = self.weights[neuronIndex][weight] * input[weight]
					output[neuron] += val
					self.inputs[neuronIndex][weight] = input[weight]

				output[neuron] += self.biases[neuronIndex]
				output[neuron] = self.__sigmoid(output[neuron])
				self.outputs[neuronIndex] = output[neuron]

				neuronIndex += 1

			input = output

		return output

	# Sigmoid function maps (-inf, inf) -> (0, 1)
	def __sigmoid(self, val):
		return 1.0 / (1.0 + math.exp(-1*val))

	# Train the network on a single set of expected vs. actual output data
	def train(self, expected, actual):
		if len(expected) != len(actual):
			sys.exit ('Provided output different size from network output.')

		# Train the output layer
		for neuron in range(0, self.numOutput):
			error = expected[neuron] - actual[neuron]
			neuronIndex = self.__getIndex(self.numLayers-1) + neuron
			self.__trainNeuron (neuronIndex, error)

		# Train the hidden layers
		for layer in range(self.numLayers-2, 0, -1):
			numNeurons = self.neurons[layer]

			for neuron in range (0, numNeurons):
				neuronIndex = neuron + self.__getIndex(layer)
				error = 0

				for nextNeuron in range (0, self.neurons[layer+1]):
					nextNeuronIndex = nextNeuron + self.__getIndex(layer+1)
					error += (
							self.weights[nextNeuronIndex][neuron] *
							self.errorGradients[nextNeuronIndex]
							)
				self.__trainNeuron(neuronIndex, error)

		self.weights = self.tempWeights;

	# Train a neuron
	def __trainNeuron(self, index, error):
		self.errorGradients[index] = self.outputs[index]
		self.errorGradients[index] *= (1 - self.outputs[index]) * error

		numWeights = len(self.weights[index])
		for weight in range(0, numWeights):
			self.tempWeights[index][weight] += (
					self.inputs[index][weight] * self.errorGradients[index]
			)

	# Get the index of the first neuron in a layer
	def __getIndex(self, layer):
		index = 0
		for l in range(0, layer-1):
			index += self.neurons[l+1]
		return index

	# Train a neural network until the error is below the threshold
	def simulate (self, inputSet, outputSet, maxError):

		iterations = 0
		attempts = 1
		maxIterations = 100000
		maxAttempts = 5

		# Arbitrary, initial error just has to be > maxError
		error = maxError + 1

		while error > maxError:

			# Prevent the network from stalling in local mins
			if iterations == maxIterations:
				if attempts == maxAttempts:
					return False

				iterations = 0
				attempts += 1

				# Generate new weights and biases
				self.__initWeights()
				self.__initBiases()
				print('Network failed to converge. Trying again with new vals')

			error = 0

			# Start at a random index to prevent getting stalled
			startIndex = random.randrange(0, len(inputSet))

			# Train on each of the input/output data sets
			for i in range (0, len(inputSet)):
				index = (startIndex + i) % len(inputSet)
				output = self.eval(inputSet[index])

				# Sum-squared error
				error += math.pow(self.__maxDiff(outputSet[index], output), 2)

				# Train the neural network
				self.train(outputSet[index], output)

			iterations += 1

		# Network converged
		return True

	# Find the maximum difference between two numeric arrays
	def __maxDiff (self, alist, blist):
		if len(alist) != len(blist):
			sys.exit('Lists must be of same size!')

		max = None
		for i in range (0, len(alist)):
			dif = alist[i] - blist[i]
			if max == None or max < dif:
				max = dif

		return max

# Convert a list of values two a 2D list
# Each line is one element of the list
def fileToList(path):
	f = open(path, 'r')
	string = None
	list = []

	for line in f:
		strArr = line.split()
		valArr = [float(strArr[i]) for i in range(0, len(strArr))]
		list.append(valArr)

	return list

def _initParsers():
	import argparse

	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest='command')

	# Create a new neural network
	parser_new = subparsers.add_parser(
		'new',
		help='Create a new neural net.'
	)
	parser_new.add_argument(
		'filepath', 
		type=str, 
		nargs=1, 
		help='Filepath to which the net should be saved.'
	)
	parser_new.add_argument(
		'neurons', 
		type=int,
		nargs='+',
		help='Number of neurons in each layer. (Must have at least 2 layers!)'
	)

	# Train a neural network
	parser_train = subparsers.add_parser(
		'train', 
		help='Train a neural net on a set of input/output data'
	)
	parser_train.add_argument(
		'filepath',
		type=str,
		nargs=1,
		help='Filepath to the neural network.'
	)
	parser_train.add_argument(
		'input',
		type=str,
		nargs=1,
		help='File path to the training input.')
	parser_train.add_argument(
		'output',
		type=str,
		nargs=1,
		help='File path to the training output.'
	)
	parser_train.add_argument(
		'-e', '--maxError',
		type=float,
		nargs=1,
		help='The desired accuracy of the network'
	)

	# Evaluate input data on the network
	parser_eval = subparsers.add_parser(
		'eval',
		help='Evaluate input/output data on the neural network'
	)
	parser_eval.add_argument(
		'filepath',
		type=str,
		nargs=1,
		help='Filepath to the neural network.'
	)
	group = parser_eval.add_mutually_exclusive_group(required=True)
	group.add_argument(
		'-i', '--input',
		type=float,
		nargs='+',
		help='A single set of input values'
	)
	group.add_argument(
		'-l', '--loadInput',
		type=str,
		nargs=1,
		help='A file from which to load input data'
	)

	return parser


# Commandline usage
if __name__ == "__main__":

	parser = _initParsers()
	args = parser.parse_args()

	command = args.command.split()[0]

	# User wants to create a new neural net
	if command == 'new':
		numLayers = len(args.neurons)
		if numLayers < 2:
			sys.exit('Error: Must have at least 2 layers')

		net = NeuralNetwork(args.neurons)
		net.save(args.filepath[0])
		print('Neural network created and saved to ' + args.filepath[0] + '.')

	# User wants to train a neural net
	elif command == 'train':
		net = NeuralNetwork.load(args.filepath[0])
		print('Neural network loaded. ' + str(net.numLayers) + ' layers.')

		inputSet = fileToList(args.input[0])
		outputSet = fileToList(args.output[0])

		print('Beginning to train')

		if args.maxError:
			maxError = args.maxError[0]
		else:
			maxError = 0.01

		net.simulate(inputSet, outputSet, maxError)
		net.save(args.filepath[0])

	# User wants to evaluate some input on the neural net
	elif command == 'eval':
		net = NeuralNetwork.load(args.filepath[0])

		if args.input:
			print(net.eval(args.input))
			sys.exit()

		input = fileToList(args.loadInput[0])
		for i in input:
			print(net.eval(i))

