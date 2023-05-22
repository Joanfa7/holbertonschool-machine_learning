# Convolutional Neural Network

1.  What is a convolutional layer?

    - A convolution layer is a fundamental building block of convoltional Neural Newtworks. It perfomrs teh convolution operation, wich involves applying a set of learnable filters to an input data values (or tensor) to extract spatial paterns or features.

    In the context of neral language processing (NLP), a convolutional layer can be applied to textual data by treating it as a one-dimensional sequence of word embeddings or characters. The fiters slide over the input, capturing local patterns and creatign feature maps. These feature maps represent the presence of specific freatures of pattersn at different locations in the input sequence.

    By using multiple filters, a convolutional layer can learn to extract various types of features form the input, such as n-grams of local context information. This allows CNNs to capture different levels of abstraction and hierichical representation of the text.

    After teh convolution operation, typical operations such as activation function (e.g. ReLU), pooling (e.g. max pooling), and possibly addicional layers like fully connected layers may be applied to further process the extracted features.

2.  What is a pooling layer?
    A pooling layer in a Convolutional Neural Network (CNN) is a layer that perfomrs a downsapmling operation along the spatial dimensions (width, height). The main objective of a pooling layer is to reduce the spatial size of a representation, thus reducing the amount of parameters and computation in the network.

    There are several types of pooling, but the most common ones are:

        - Max Pooling
        - Avarage Pooling
        - Global Pooling

    The pooling operation is important for achiving translation invariance in CNN. It allows the network to etect a feature regardless of where it is located in the input.

3.  Forward propagation over convolutional and pooling layers

    - Forward propagation is the process by which information moves thwough a neural network, with each layer perfomring a set of operation on the input data and passing the result to the next layer.

    Let's go through the steps in forward propagation for both convolutional and pooling layers:

        1. Convolutional Layers:

            - Input: This layer accepts a volume of data. for the first convolutional alyer in a network,this is your input image. For subsequent layers, it's the output form the precious layer.

            - Filters: Each filter, or kernel, in the layer is convolved across the width and height of the input volume. this process involves across the width and height of the input volume. This process involves elementwise multiplication of the filter with the part of the image it's currently on, abd then summing up the results int a single output pixel. This operation is performed for al depth cannels (e.g., in an RGB image, there would e three depth chanels: red, gree, and blue)

            3. Stride and Padding: The stride controls how the filter moves across teh input values. A stride of 1 moves the filter one pixel at a time. Padding involves adding a border of zero around the input volume. This allows the convolutional layer to control the spatial dimensions of the output volume.

            4. Activation Function: After the convolution operation, an activation funciton like ReLU is typically applied alement-wise. It introduces non-linearty into the model, allowing it to learn more complex features.

        2. Pooling Layer:

            1. Input: This layer accepts the output from a previous layer.

            2. Pooling Operation: The layer performs a downsampling operation along the spatial dimentions of the input.

            3. Stride: Like in the convoluitonal layer, the stride contrls how the pooling operaiton moves across teh input

        It's important to note that inlike other layers in a Neural Network, pooling layers don't have any weights that need to be learned during training. Their behavior is defined purely by their hyperparameters (like the pooling operations and teh strde.)

4.  Back propagation over convolutional and pooling layers:

    - Forward propagation is the process by which information moves through a Neural Network, with each layer performing a set of operations on the input data nad passing the results to the next layer.

    Convolutional Layer: 1. Input: This layer accepts a volume of data. For the firts convolutional layer in a network, this is you input image. for subsequent layers, it;s the output from the precious layer.

        2. Filters: Each filter, or kernel, in the layer is convolved across the width and the height of the input volume. This process involves element- wise multiplication of teh filter with the part of the image it's currently on, and then summing up the results into a single output pixel. This operation is perfomred for

5.  How to build a CNN using Tensorflow and Keras
