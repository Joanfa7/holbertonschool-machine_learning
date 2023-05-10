# Keras

![keras](https://keras.io/img/logo.png)

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

          from keras.models import Sequential
          from keras.layers import Dense

          model Sequential()
          model.add(Dense(64, activation='relu', input_shape=(784,)))
          model.add(Dense(32, activation='relu'))
          model.add(Dense(10, activation='softmax'))

    - The Functional API allows for more flexibility when defining the model's architecture. You can create models with multiple inputs, multple outputs, shared layers, or overn recurrent connections between layers. this approach is suitable for more complex neural network architectures.

              from keras.layer import Input, Dense
              from keras. models import Models

              inputs = Input(shape=(784,))
              x = Dense(64, activation='relu')(inputs)
              x = Dense(32, activation='relu')(x)
              outputs = Dense(10, activation='softmax')(x)

              model = Model(inputs= inputs, outputs= outputs)

      Both APIs can be used to create a wide cariaty of neural newtork architectures in Keras, but the coice depends on the complexity of the model and the specific requirements of your project.

4.  How to build a layer
    In Keras, you build layers using predefined classes. Each type of layer serves a different purpose, such as dense layers for fully connected networks, convolutional layers for image processing, and recurrent layers for sequences. When creatign a layer, you typically provide some parameters, such as the number of neurons or the activation function.

    Example:

    1.  Dense (Fully Connected) Layer:

            from keras.layers import Dense

            dense_layer = Dense(128, activation='relu')

            # Creates a dense layer with 128 neurons using the ReLU activation funciton.

    2.  Convolutional Layer:

            from keras.layers import Conv2D

            conv_layer = Conv2D(32, kernel_size(3, 3,), activation 'relu')

    3.  Recurrent Layer (LSTM):

        from keras.layers import LSTM

        lstm_layr = LSTM(64) # This creates a recurrent LSTM layer with 64 units.

5.  How to add regularization to a layer
    Regularization is a technique used to prevent overfitting in neural networks by adding a penalty to the loss function based on the complexity of the model. In Keras, you can be add regularization to layers using the 'hernel_regularizer', 'bias_regularizer', and 'activity_regularizer' arguments when creating a layer.

    1. L1 regularization:

       from keras.layers import Dense
       from keras.regurlarizers import L1

       dense_layer = Dense(128, activation='relu', kernel_regularizer=L1(l1=0.01))

       #This creates a dense layer with 128 neurons, ReLU activation, and L1 regularization on hte weights with a regularization strength of 0.01

    2. L1_L2 regularization:

       from keras.layers import Dense
       from keras.regularizers import L1L2

       dense_layer = Dense(128, activation='relu', kernel_regularizer=L1L2(L1=0.01, l2=0.01)) # This creates a dense layer with 128 neurons, ReLU activation,and both L1 and L2 regularization on the withs with a regularizaiton stregth of 0.01 for each.

    You can use the same approach to add regularization to other types of layers, such as convolutional an recurrent layers.

6.  How to add dropout to a layer

    In Keras, adding a dropout layer to your neural network is straight forward. The 'Dropout' layer is used to apply dropout regularization during training, which helps prevent overfitting by randomly setting a fraction of the input units to 0 at each update.

        from keras.models import Sequential
        from keras.layers import Dense, Dropout

        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(784,)))
        model.add(Dropout(0.5)) # Add a dropout layer with a dropout rate of 0.5 (50%)
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='softmax))

    In this example, we first create 'Sequiential' model and add a dense layer with 128 neurons and the ReLU acrivation function. Ather de dense layer, we add a 'Droput' layer with a doropout rate of 0.5 (50%). This means that during the training, approximately 50% of the input neurons will be randomly set to 0 at each update. This helps the to prevent overfitting by reductin the model's reliance on specific neurons.

    The dropout layer can be added after any type of layer where dropout regularization is desired, such as dense, convolutional, or recurrent layers. The dropout rate parameter (a value between 0 to 1) controls the fraction of input units that will be set to 0, and it can be adjusted based on your specific problem and dataset.

7.  How to add batch normalization

    Batch Normalization is a technique often used to speed up training in deep neural networks. It normalizes the activations of the precious layer at each batch by applying a transormation that maintains the mean activation close to 0 and the activation standard deviation closes to 1.

        from keras.models import Sequential
        from keras.layers import Dense, BatchNormalization

        model = Sequential()
        model.add(Dense(128, activation='relu', input_shap=(748,)))
        model.add(BatchNormalizatioin())
        model.add(Dense(64, activation=relu))
        model.add(Dense(10, activation='softmax'))

    In this example,After the dense layer, we add a 'BatchNormalization' layer. The 'BatchNormalization' layer. The 'BatchNormalization' layer will normalize the activations of the provious layer at each batch.

    You can add a 'BatchNormalization' layer after any type of layer where you want the activations to be normalized. Batch Normalization can provide several benefits, such as speeding up training, requiring less careful initialization, and sometimes even eliminating the need for Dropout.

8.  How to compile a model

    In Keras, compiling a model requires specifiying the loss function, the optimizer, and the metrics for evaluation. This is done using the 'compile' method of the model.

        from keras.model import Sequential
        from keras.layer import Dense
        from keras.optimizers import Adam

        # define the model

        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(748,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(10, activation='softmax))

        # model.compile(optimizer=Adam(),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    - 'optimizer='Adam()': This specifies the optimization algorithm to use. Adam is a comonly used optimizer that adjusts the learning rate adaptively.

    - 'loss='categorizal_crossentropy'': This specifies the loss functio to use. The categorical cross-entropy loss function is commonly used for multi-class classification problems

    - 'metrics=\['accuracy']: This specifies the metrics to ecaluate the model during training and testing. Here we use accuracy, which is a common metric from classification problems.

    Note that the appropriate loss function and metrics will be depend on your specific problem and the type of output your model is designed to predict. Fro instance, for a binary classification problem, you migh use 'binary_crossentropy' as the loss function. For a regression problem, you might use 'mean_sequared_error'/

9.  How to optimize a model

    Optimizing a machine learning model involves several stategies and techniques. Here are some of the most common approaches: 1. Choose the richt Architecture: Depending on the problem you're solvin (classification,regression, etc.) certain architectures may perfomr better than others. For example, Convolutional Neural Networks (CNNs) generally perform well on image data, wile Recurrent Neural Networks (RNNs) are often used for sequential data.

        1. Hyperparameter Tuning: Adjusting the parameters of the learning algorithm can have a big impact on model performance. This includes things like the learning rate, batch size, number of layers, number of neurons per layer, and dropout rate. Grid search or randomized search can be used for exhaustive search, while Bayesian optimization, generic algorithms, or thechniques like optuna can be used for more efficient search.

        2. data Augmentation: By artificially increasing the size of the training set, data augmentation can help improve performance and prevent overfitting. This can involve things like rotating, shifting, or flipping images for a convolutional neural network.

        3. Early Stopping: This technique stops training when te perfomrance on a validation set stps improving, which can prevent overfitting.

        4. Batch Normalization: Batch normalization standardizes the imputs to a layer for each mini-batch, which can help improve the speed, performance, and stability of the neural network.

    The process of optimizing a model involves a lot of experimentation and iteration, and the techinques that work best can vary depending on the specific problem and dataset.

10. How to fit a model
    Fitting a model in keras involves providing the training data and labels to the model and specifying the number of epochs and the batch size. This is done using the 'fit' methd of the model.

        # asume we have some training data and labels
        train_data = ...
        train_labels = ...

        # Compile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy'
                      metrics=['accuracy'])

        # fit the model

        model.fit(train_data, train_labels, epochs=10, batch_size=32)


        1. 'train_data': This is the training data that will be used to train the model.

        2. 'train_label': These are the labels corresponding thetraining data.

        3. 'epoch=10': This specifies the number of times the learning algorithm will pass thwough the entire training dataset. One epoch means one pass though the entire dataset.

        4. 'batch_size=32': This trains the model for a fixed number of epochs.

    You can also specify a validation set to monitor the validation loss and metrics during training:

        # Assume we have some validation data and labels
        val_data = ...
        val_labels = ...

        model.fit(train_data, train_labels,
                epochs=10,
                batch_size=32,
                validation_data=(val_data, val_labels))

11. How to use validation data
    Validation data is a subset of your training data that you use to ecaluate the perfomrance of your model during training. This allows you to see hwo well you model generalizes to unseen data and can help diagnose issues like overfittig

    To use validation data in Keras, you can provide it as an argument to the 'fit' method using the 'calidation_data' parameter.

        history = model.fit(x_train, y_train,
                            epochs=10,
                            batch_size=32,
                            validation_data=(x_val, y_val))

    In this example, 'x_train' and 'y_train' are the features and labels for the trainig data, while 'x_val' and 'y_val' are the features and labels for the calidation data.

    Alternatively, you can use the 'validation_split' argument to automatically reverse a portion of the training data for validation

        history = model.fit(x_train, y_train,
                            epochs=10,
                            batch_size=32,
                            validation_split=0.2)

    In this case, Keras will use the last 20% of the data for validation.

    During training, Keras will outpu the loss and accuracy (or whatever metrics you chose when compilling the model) on the validation data.

    You can use the 'history' object returned by the 'fit' method to plot the training and validation loss and accuracy over time, which can help you idagnose issues like overfitting

12. How to perform early stopping
    early stopping is a technique to prevent a overfitting by stopping the training process once the model perfomrance stpos improving on a hold out validation dataset.

        from keras.callbacks import earlyStopping

        # define the early stopping callback
        earlystop_callback = EarlyStopping(
            monitor='val_loss', #monitor the validation loss
            patience=3, # number of epochs to wait befor tropping
            restore_best_weights=True) # restore the best height from training

        # fit the model
        history = model.fit(
            x_train, y_train,
            epoch=100, # maximum number of epochs to run
            validation_data = (x_val, y_val),
            callbacks=[earlystop_callback]) #pass the early stopping callback

    We monitor the validation loss('cal_loss') and stop training if it hasn't decreased for 3 epochs (as defined by the 'patience' parameter). The 'restore_best_weights' option rolls back the model weights to the state at the epoch with the best monitored matric values ('val_loss', in this case.)

13. How to measure accuracy
    Measuring accuracy in queras is quite straightforward. When you compile your model, you can specify 'accuracy' as one of the metrics that you want to calculate during training"

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    Then, during the training (e.i. when calling 'fit' method), Keras will output the accuracy form each epoch both for the training and calidation data, if provided

        history = model.fit(x_train, y_train,
                            epochs = 10,
                            validation_data=(x_val, y_val))

    After training, you can also evaluate the model's accuracy on a test set using the 'evaluate' method.

        loss, accuracy = model.evaluate(x_test, y_test)

    'x_test' and 'y_test' are your test features and labels, The 'evaluate' method returns the loss and accuracy (because we specified 'accuracy' as a metric when we compiled the model)

14. How to make a prediction with a model
    Once you've trained your model, you can use it to make predictions on new data using the 'predict' method in Keras.

        # assuming that you have new data in x_new
        predictions = model.predict(x_new)

15. How to access the weights/outputs of a model
    In Keras, you can access the weights of a model using the 'get_weights' method and the outputs of a model usign the 'output' property.

        1. Accessing Weights: You can use the 'get_weights' and 'set_weights' methods to ge and set the model weights. Each layer has these methods which return or set the weights for that layer.

            # get the weights of the first layer
            weights = model.layers[0].get_weights()

            # set the weights of the first layer
            model.layer[0].set_weigts(weights)


        Note that 'get_weights' retuns a list of numpy arrays. For a Dense layer, the list will have two elements: the first element is a 2D array of the weights, and the second element is a 1D array of the biases.

        2. Accessing Outputs: Each layer has an 'output' property that gices the symbolic output of that layer. Here's how you can get the output of a specific layer:


            # get the ouput of the first layer
            outpu = model.layers[0].output


        If you want to get the actual values of a layer for a specific input, you can define a new model that ends at the layer and call 'predict' on it:


            from keras.models import Model

            # define a new model that outpust the outputs of the first layer.
            intermediate_model = Model(inputs=model.input, outputs=model.layers[0].output)

            # get the output values of the first layer for a sepcific input.
            intermediate_output = intermediate_model.predict(x_test)


        In this example, x_test is the input data for which you want to get the output of the first layer. The 'predict' method returns the actual output values of the first layer for 'x_test'.
