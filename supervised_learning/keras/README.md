# Keras

!(Keras)[https://keras.io/img/logo.png]

## Learning Objectives

1.  What is Keras?
    Is an open-source, high-level neural netwokr library written in python. It is designed to make it easy for users to build and experimetn with deep learning models. It provides a simple, intuitive way to define treain and evaluate nerutal networkds by offering essential building blocks called layers, wich can be combined o create a variety of architectures. The library supports carious types of layers, such as dense (fully connected), convolutional, recurrent and more.

2.  What is a model?
    A model is a mathematical representation of a real-world preoblem or system, designed to learn patterns form data. The model is created using algorithms and techniques that allow it to identify underlying relationships and structures in the data. Once trainde, de model can make predictions, classify data, or generate new data based on the patterns it has learned.

    There are various types of models in machine Learning, such as linear regression, decision trees, support vector machines, and neural networks. In deep learning, the most common models are neural networks, wich concist of interconnected layers of neurons ( artificial nodes that imitate biological neurones). These models are3 designed to automatically learn complex features and representations form large datasets.

    A typical deep learning model consists of the following components:

        - Architecture: The structure of the model, including the number of layers, the types of layers(e.g., dense, convolutional, recurrent), and the connections between them.

        - Parameters: The weights and biases with the model, wich are adjusted during training to minimize the difference between the model's predictions and the actual output (i.e., to reduce the loss function)

        - Activation functions : Nonlinear functions applied to the output of each reuton that determine the final output. Some common activation functions include the sigmoid, ReLU (rectified linear unit), and softmax functions.

        - Loss function: A mentric used to meassure the difference between the model's predictions and the actual output. the goal training is to minimize the loss function. Common loss function include mean squared error, categorical cross-entropy, and binary cross-entropy.

        -Optimization: An algorithm used to update the model's parameters during training, such as gradient descent or its cariations like strochatic gradient descent, Adam, and RMSprop.

3.  How to instantiate a model (2 ways)

    In Keras, you can create a neural network model using two different approaches: the Sequentail API and the Functional API.

        - The Sequential API is a lineal track of layers, where you simpli add one layer at a time, and each layer is connected to the precious one in the order they are added. This approach is suitable for simple feedforward neural networks with any complex connections or braching.

        ```
            from keras.models import Sequential
            from keras.layers import Dense

            model Sequential()
            model.add(Dense(64, activation='relu', input_shape=(784,)))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(10, activation='softmax'))

        ```

        - The Functional API allows for more flexibility when defining the model's architecture. You can create models with multiple inputs, multple outputs, shared layers, or overn recurrent connections between layers. this approach is suitable for more complex neural network architectures.

        ```
            from keras.layer import Input, Dense
            from keras. models import Models

            inputs = Input(shape=(784,))
            x = Dense(64, activation='relu')(inputs)
            x = Dense(32, activation='relu')(x)
            outputs = Dense(10, activation='softmax')(x)

            model = Model(inputs= inputs, outputs= outputs)
        ```

4.  How to build a layer
5.  How to add regularization to a layer
6.  How to add dropout to a layer
7.  How to add batch normalization
8.  How to compile a model
9.  How to optimize a model
10. How to fit a model
11. How to use validation data
12. How to perform early stopping
13. How to measure accuracy
14. How to evaluate a model
15. How to make a prediction with a model
16. How to access the weights/outputs of a model
17. What is HDF5?
18. How to save and load a model’s weights, a model’s configuration, and the entire model
