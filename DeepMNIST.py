import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Create input object which reads data from MNIST datasets.  Perform one-hot encording to define the digit.
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# Using Interactive session makes it the default session, so we do not need to pass sess for commands
sess = tf.InteractiveSession()

# Define the placeholders for MNIST input data
x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# change the MNIST input data from a list of values to a 28 pixel X 28 pixel X 1 grayscale value cube
# which the Convolution NN can use.
x_image = tf.reshape(x, [-1, 28, 28, 1], name = "x_image")


# Define helper functions to created weights and bias variables, convolution and pooling layers.
# We are using RELU as our activation function.  These must be initialized to a small positive number
# and with some noise so you don't end up going to zero when comparing diffs
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# Convolution and Pooling - we do Convolution, and then pooling to control overfitting
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

# Define layers in the NN (Neural Network)

# 1st Convolution layer
# 32 features for each 5x5 patch of the image
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# Do convolution on images, add bias and push through the 
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# take results and run through max_pool
hpool1 = max_pool_2x2(h_conv1)

# 2nd Convolution layer
# Process the 32 features from Convolution layer 1, in 5 x 5 patch.  Return 64 features weights and biases
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([32])
# Do convolution of the output of the 1st convolution layer.  Pool results
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
