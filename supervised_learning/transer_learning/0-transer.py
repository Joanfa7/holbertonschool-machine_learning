#!/usr/bin/env python3
''' Transfer Learning
    training a convolutional neural network to classify the CIFAR 10 dataset
    https://www.cs.toronto.edu/~kriz/cifar.html'''


import tensorflow.keras as K



def preprocess_data(X, Y):
    ''' pre-processes the data for your model
        X: numpy.ndarray (m, 32, 32, 3) CIFAR 10 data
            m: number of data points
        Y: numpy.ndarray (m,) CIFAR 10 labels for X
        Returns: X_p, Y_p
            X_p: preprocessed X
            Y_p: preprocessed Y'''
    X_p = K.applications.resnet.preprocess_input(X)# resnet50
    Y_p = K.utils.to_categorical(Y, 10)# one-hot encode

    return X_p, Y_p # preprocessed X, preprocessed Y


if __name__ == '__main__':


    (x_train, x_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_p, Y_p = preprocess_data(X_test, Y_test)
    x_test, y_test = preprocess_data(x_train, Y_train)

    resnet_base = K.applications.ResNet50(include_top=False, input_shape=(224, 224, 3)
                                    )
    
    input_layer = K.Input(shape=(32, 32, 3))
    resize_layer = K.layers.Lambda(lambda image: K.preprossesing(image.smart_resize(image, (224, 224))))(input_layer)
    
    resnet_base_layer = resnet_base(resize_layer, training=False)
    flat_layer = K.layers.Flatten()(resnet_base_layer)
    dense_layer_1 = K.layers.Dense(500, activation='relu')(flat_layer)
    dropout_layer_1 = K.layers.Dropout(0.3)(dense_layer_1)
    final_output_layer =  K.layers.Dense(10, activation='softmax')(dropout_layer_1)
    model = K.Model(inputs=input_layer, outputs=final_output_layer)

    model.summary()

    resnet_base.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1, validation_data=(X_test, Y_test))

    output = model.evaluate(X_p, Y_p, batch_size=128, verbose=1)

    model.save('cifar10.h5')





