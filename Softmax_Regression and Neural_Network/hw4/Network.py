# Standard library
import random
from cv2 import REDUCE_SUM

# Third-party libraries
import numpy as np


import multiprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
# from IPython.display import set_matplotlib_formats
# %matplotlib inline

class Network(object):
	def __init__(self,sizes): #the list sizes contains the number of neurons in the respective layers.
		self.num_layers = len(sizes)  #the number of the layers in Network
		self.sizes = sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]
		self.weights = [np.random.randn(y,x) 
						for x,y in zip(sizes[:-1],sizes[1:])]

	def feedforward(self,a):
		"""Return the output of the network if "a" is input"""
		for b,w in zip(self.biases,self.weights):
			a = sigmoid(np.dot(w,a) + b)		
		return a
		
	# def SGD(self, training_data, epochs, mini_batch_size,eta,
	#         test_data = None):
	# 	"""
	# 	Train the neural network using mini-batch stochastic gradient descent.
	# 	The "training_data" is a list of tuples "(x,y)" representing the training inputs
	# 	and the desired output. The other non-optional parameters are self-explanatory.
	# 	If "test_data" is provided then the network will be evaluated against the test 
	# 	data after each epoch, and partial progress printed out. This is useful for tracking
	# 	process, but slows things down substantially.
	# 	"""
	# 	if test_data: 
	# 		n_test = len(test_data)
	# 	n = len(training_data)
	# 	for j in range(epochs):
	# 		random.shuffle(training_data)        #rearrange the training_data randomly
	# 		mini_batches = [ training_data[k:k + mini_batch_size]
	# 		                               for k in range(0, n, mini_batch_size)]
	# 		for mini_batch in mini_batches:
	# 			self.update_mini_batch(mini_batch,eta)
	# 		if test_data:
	# 			print("Epoch {0}: {1} / {2}".format(j,self.evaluate(test_data),n_test))
	# 		else:
	# 			print("Epoch {0} complete".format(j))
	def SGD(self, training_data, epochs, mini_batch_size,eta,
	        test_data = None):
		"""
		Train the neural network using mini-batch stochastic gradient descent.
		The "training_data" is a list of tuples "(x,y)" representing the training inputs
		and the desired output. The other non-optional parameters are self-explanatory.
		If "test_data" is provided then the network will be evaluated against the test 
		data after each epoch, and partial progress printed out. This is useful for tracking
		process, but slows things down substantially.
		"""
		#if test_data: 
		n_test = len(test_data)
		training_accuracy = []
		testing_accuracy = []
		training_loss = []
		testing_loss = []

		n = len(training_data)
		for j in range(epochs):
			random.shuffle(training_data)        #rearrange the training_data randomly
			mini_batches = [ training_data[k:k + mini_batch_size]
			                               for k in range(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,eta)
			
			entropy_1,accuracy_1 = self.cross_entropy(training_data)
			entropy_2,accuracy_2 = self.cross_entropy(test_data)
            
			training_accuracy.append(accuracy_1)
			training_loss.append(entropy_1)
			testing_accuracy.append(accuracy_2)
			testing_loss.append(entropy_2)

			print("Epoch %s, Training_acc %s, testing_accuracy %s, Train loss: %s, Test Loss: %s" % (j, accuracy_1, accuracy_2, entropy_1, entropy_2))

		return training_accuracy,testing_accuracy,training_loss,testing_loss
			#if test_data:
			#	print("Epoch {0}: {1} / {2}".format(j,self.evaluate(test_data),n_test))
			#else:
			#	print("Epoch {0} complete".format(j))
	


	def cross_entropy(self,data):

		results = [(self.feedforward(x),y) for (x,y) in data]

		accuracy = 0	
		entropy = 0
		for (x,y) in results:
			if type(y) is np.int64:
				if np.argmax(x) == y:
					accuracy += 1
				entropy += -np.log(x[y])
			else:
				if np.argmax(x) == np.argmax(y):
					accuracy += 1
				entropy += - np.log(x[np.argmax(y)])
		
		accuracy = accuracy/len(data)
		entropy = entropy/len(data)
		return entropy[0],accuracy

			
	def update_mini_batch(self,mini_batch,eta):
		"""
		Update the network's weights and biases by applying gradient descent using backpropagation
		to a single mini batch. The "mini_batch" is a list of tuples "(x,y)" and "eta" is the learning
		rate.
		"""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]
	
	def backprop(self, x, y):
		"""Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
		activation = x
		activations = [x] # list to store all the activations, layer by layer
		zs = [] # list to store all the z vectors, layer by layer
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
        # backward pass
		delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			change = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = change
			nabla_w[-l] = np.dot(change, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def evaluate(self, test_data):
		"""Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
		test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

	def cost_derivative(self, output_activations, y):
		"""Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
		return (output_activations-y)


		
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

import sys
sys.path.append("../")

# import loader
# training_data, validation_data, test_data = loader.load_data_wrapper()
# net = Network([784, 30, 10])
# train_acc, net.SGD(training_data, 2, 10, 3.0, test_data=test_data)




def plot_result(training_loss,testing_loss,name,title):
    from matplotlib.legend_handler import HandlerLine2D
    epoch = [i for i in range(1,len(training_loss)+1)]
    line1, = plt.plot(epoch, training_loss, 'b', label='Train ' + name)
    line2, = plt.plot(epoch, testing_loss, 'r', label='Test ' + name)
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel(name)
    plt.xlabel('Epoch')
    plt.title(title)
    plt.show()

from sklearn.metrics import confusion_matrix
import itertools
def cm_plot(predictions,label):
    
    classes= ["0", "1", "2", "3", "4", "5","6","7","8","9"]
    
    predictions = np.array(predictions)
    y_test = np.array(label)
    
    cm = confusion_matrix(y_test,predictions)
    normalize=False
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.title('MNIST Classification')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    width = 40
    height = 40
    plt.figure(figsize=(width, height))
    plt.show()


import loader
def main():
    training_data, validation_data, test_data = loader.load_data_wrapper()
    net = Network([784, 50, 10])
    training_accuracy,testing_accuracy,training_loss,testing_loss = net.SGD(training_data, 20, 50, 2.0, test_data=test_data)
    
    plot_result(training_loss,testing_loss,"Cross Entropy Loss","Training and Test Loss")
    plot_result(training_accuracy,testing_accuracy,"Accuracy","Accuracy Plot")
    
    prediction = []
    label = []
    for (x,y) in validation_data:
        prediction.append(np.argmax(net.feedforward(x)))
        label.append(y)
        
    cm_plot(predict,label)
    val = 0
    for i in range(len(label)):
        if label[i] == predict[i]:
            val += 1
    print("Validation Preidction: {0} / {1}".format(val,len(prediction)))



if __name__ == '__main__':
    main()