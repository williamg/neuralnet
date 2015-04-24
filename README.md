neuralnet
===
A Python utility for using feed-forward artificial neural networks

 - [Command Line Usage][1]
 - [NeuralNet Class Reference][2]

---
<a name="cl"></a> Command Line Usage
---

The utility can be run from the command line as follows:

    python neuralnet.py [-h] {new,train,eval} ...
    
There are three different commands that can be used to create new networks, train them, and use them to evaluate data.

**Creating a new network:**

    python neuralnet.py new /path/to/save/network.tnxt 2 3 1
    
The above command will create a new network with 3 layers consisting of 2 neurons, 3 neurons, and 1 neuron, respectively, and save it to the designated path.

**Training a network:**

    python neuralnet.py train /path/to/saved/network.txt trainingInput.txt trainingOutput.txt
    
The above command will load the network saved at the designated path and train it on the provided input and output data. The input and output data should consist of values separated by a single space with each line representing a different set of training data. The first line of the input data should correspond to the first line of the output data, and so on.

An optional parameter can be specified with `-e MAXERROR` where `MAXERROR` is the desired accuracy of the network (no error exceeds the provided value).

The trained network will be saved to the path from which the network was originally loaded.

**Evaluating data on the network:**  
Data can be evaluated on the network in two ways:

    python neuralnet.py eval -i 1 0 /path/to/network.txt
    
Will evaluate the dataset `1 0` on the neural net and report the output.

    python neuralnet.py eval -l input.txt /path/to/network.txt
    
Will evaluate all datasets contained in the input file and report the output.

<a name="cr"></a> NeuralNet Class Reference
---
A fully-functional NeuralNet class is also included that can be imported into Python projects.

**Initializing the class:**  
The neural network can be initialized with random weights and biases as follows:

    neurons = [2, 3, 1]
    nn = NeuralNetwork(neurons)
    
Alternatively, if you have a list of weights and biases, the neural network can be initialized as follows:

    nn = NeuralNetwork(neurons, weights, biases)
    
The `weights` array is a two-dimensional list. `weights[i][j]` corresponds to the weight between the neuron `i` and the `j'th` neuron in the previous layer.

The `biases` array is simply a list of biases for each neuron.

**Saving a network:**  
A neural network can be saved using the `save` function:

    nn.save('/path/to/save.txt')
    
Will save the neural network to the provided path.

**Loading a network:**  
A neural network can be loaded using the `load` function:

    nn = NeuralNetwork.load('/path/to/load.txt')

**Training the network:**  
The network can be trained on a set of data as follows:

    input = [[0, 0],[0,1],[1,0],[1,1]]
    output = [[0],[1],[1],[0]]
    maxError = 0.01
    nn.simulate(input, output, maxError)

**Evaluating data on a network:**  
To evaluate a set of data on the network, use the `eval` function:

    input = [1,0]
    output = nn.eval(input)
    
*Note that when training the network, a list of data sets is expected while one data set is expected when evaluating data*

---
Developed by William Ganucheau. Released under the MIT License.


  [1]: #cl
  [2]: #cr
