# Perceptron, Neural Networks, and Regularization
Perceptron is a single-layer neural network used for binary classification tasks in machine learning. The perceptron algorithm works by taking in inputs, which are multiplied by weights, and then a bias term is added. The resulting output is passed through an activation function, which determines the final output. The perceptron algorithm updates the weights and bias during the training process to minimize the error.

Neural networks are a type of machine learning algorithm used for both regression and classification tasks. They are composed of layers of interconnected nodes called neurons, where each neuron takes inputs, applies a transformation, and then passes the output to the next layer. The output of the final layer corresponds to the predicted output.

Regularization is a technique used in machine learning to prevent overfitting and improve the generalization performance of models. It works by adding a penalty term to the loss function, which encourages the model to use simpler solutions. The two most common types of regularization are L1 and L2 regularization, which add the absolute value and square of the weights, respectively, as the penalty term.

Neural networks and perceptrons can both benefit from regularization to prevent overfitting and improve the generalization performance of the models. Regularization can also be used with other types of machine learning algorithms, such as linear regression and logistic regression, to improve their performance.

## Coding
- Implement the Perceptron classifier
- Write the `forward`, `backward`, and `fit` functions to enable training
  a MLP written only in `numpy`.
- Implement squared error loss function
- Implement the [ReLU activation
function](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
- Implement regularization for the MLP's weights
- Create a feature transformation that allows a linear model to classify
  a challenging spiral dataset.
