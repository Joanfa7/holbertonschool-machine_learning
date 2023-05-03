# Regularization

1.  What is regularization? What is its purpose?

    - Regularization is a technique used in machine learning to prevent overfitting by adding a penalty term to the loss function. Its purpose is to encourage simpler models that generalize better to new, unseen data by discuraging excessively large parameter values or overly complex models.

2.  What is are L1 and L2 regularization? What is the difference between the two methods?

    - L1 and L2 regularization are two common types of regularization:
      - L1 regularization (also known as Lasso regularization) adds the absolute value of the model's parameters (coefficients) multipled by a regularization parameter (lambda) to the loss function. This encourages sparsity in the parameters values, effectively performing feature selection.
      - L2 regularization (also known as Ridge regularization) adds the square of the model's parameters (coefficients) multiplied by a regularization parameter (lambda) to the loss function. This encourages smaller parameter values, leading to smoother models. The main difference between the two methods is that L1 regularization tends to create sparse models with few non-zero coefficients, while L2 regularization creates smoother models with many small coefficients.

3.  What is dropout?

    - Dropout is a regularization technique for neural networks that involves randomly setting a fraction of the input units to zero during training. This prevents the network from relying too heavily on individual neurons and encourages it to learn more robust represntations.

4.  What is early stopping?

    - Early stopping is a technique used to prevent overfitting by stopping the training process when the model's perfomrance on a validation set starts to degrade. This avoids overtraining the model and helps to find the optimal number of training epochs.

5.  What is data augmentation?

    - data ugmentation is a technique used to increase the size and dicersity of a training dataset by applying carious transformations to the original data. This can include rotation, scaling, flipping, and color alterations. the goal is to increase the model;s ability to generalize to new data by exposing it to more divese training examples.

6.  How do you implement the above regularization methods in Numpy? Tensorflow?

    - Numpy: you can manually implement L1 and L2 regularization by adding the corresponding penalty terms to the loss function and updating the model's parameters during gradient descent.
    - TensorFlow: You can use built-in regularization options available in the layers. For droput, use 'tf.keras.layers.Dropout'
      Early stopping can be implemented usign the 'tf.keras.callbacks.EarlyStopping', callback and data augmentation can be achived with 'rf.keras.preprocessing.image.ImageDataGenerator'

7.  What are the pros and cons of the above regularization methods?

    - L1 regularization: Pros - Encourages sparsity, performs feature selection. Cons - May result in unstable solutions when features are correlated.

    - L2 regularization: Pros - Encourages smoothness, more stable solutions. Cons - Does not perform feature selection, can be less interpretable.

    - Dropout: Pros - Efective for large neural networks, reduces overfitting. Cons: Increases training time, not applicable to all odel types.

    - Early stopping: Pros - Simple to implement, saves compuational resources. Const - Requires a separate validation set, may be sensitive to random fluctuations in performance.

    - Data augmentation: Pros - Increases dataset size and diversity, improves model generalization. Cons - Computationally exprensive, not applicable to all types of data.
