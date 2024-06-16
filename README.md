# Implementing Neural Networks with Numpy from scratch

## Logistic Regression
- folder: ```basics```
- The math behind a sigle unit of Logistic Regression is beuatifully implemented using numpy. 
- The intricacies of initializing weights and biases, forward and backward propagation, computing gradients, and updating weights and biases are implemented.
- The code is tested on various sythetic datasets and on the cat vs non cat dataset. Decision boundary and learning curves are plotted for better understanding.
<img src="sample_images/lr_output.png"/>

## Shallow Neural Networks 
- folder: ```shallow_nn```
- Neural Network using 1 hidden layer is implemented from scratch including forward and backward propagation, computing gradients, updating weights and biases.
- The number of hidden units used can be changed and the resulting model is testing on the noisy moon dataset.
<img src="sample_images/snn_output1.png"/>
<img src="sample_images/snn_output2.png"/>
<img src="sample_images/snn_output3.png"/>

## Deep Neural Networks
- folder: ```deep_nn```
- l_layer_nn is implemented from scratch. The architecture of the neural network can be provided as input and the model is trained using the given dataset.
- The intricacies of computing cost, gradients of linear and activation functions, updating weights and biases are implemented.
- This implementation is tested on the synthetic dataset and the cat vs non cat dataset. It achives a very good accuracy compared to the shallow neural network.
<img src="sample_images/lnn_output1.png"/>

## Improving Deep Neural Networks
- folder: ```improving_nn```
- improved_nn incorporates various techniques to improve the performance of the neural network such as:
    - Initialization: He initialization, Random initialization, Zero initialization
    - Regularization: L2 regularization, Dropout (Not Implemented)
    - Optimization: Mini-batch gradient descent, Momentum, Adam, RMSprop (Not Implemented)

<img src="sample_images/app_output.png"/>

## Diagnosis of Pneumonia using Deep Neural Networks trained on Chest X-ray images
- folder: ```application```
- The l_layer_nn is trained on the chest x-ray images to diagnose pneumonia.
- The current model gives very less accuracy.
- The improved neural network techniques are being implemented to improve the accuracy of the model. (Not Implemented)

## Keras Basics
- folder: ```keras_basics```
- 1. Basic Applications: Covers MNIST, Fashion MNIST, and IMDB datasets.
- 2. Operations: Covers basic operations in Keras such as Convolution, Pooling, Flatten, Dense, Dropout, BatchNormalization, etc. (In Progress)