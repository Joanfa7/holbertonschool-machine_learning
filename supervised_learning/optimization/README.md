# Optimization

# Learning Objectives

1.  What is a hyperparameter?

    - Is a parameter in machine learning model whose value is set before the learning process begins. It influences the model's performance and is not learned during training. Examples include learning rate, number of hidden layers, and batches size.

2.  How and why do you normalize your input data?

    - Normalization is the process of sacling input data to a standard range, typically \[0,1] or [-1, 1]. You normalize input data for two main reasons: 1. Improving convergence: Normalized data helps optimization algorithms, such as gradient descent, to converge faster, as it ensures freatures havee similar scales, preventing some form dominating others.

           2. Ensuring numericalstability: Some machine learning algorithms are sensitive to the scale of input features. Normalization prevents large input values form causing numerical instability or dispoportionate influence on the mode's output.

      To normalize data, common methods include Min-Max scaling and Standard (Z-score) scaling. Min-Max scaling rescales data to a range of [0, 1]
      while Standard sacling centes the data around the mean with a standard deviation of 1.

3.  What is a saddle point?

    - A saddle point is a point int he parameter space of a funciton where the gradient is zero (i.e., all partial derivatives are zero), but it is neither a local minimum nor a local maximum. Insead, it's a point where the function has a mix of concave and convex curvature in different dimensions. In the contex of optimization, saddle points can be problematic, as they may cause gradient-based optimization algorithms to stall or converge slowly, especially in high-dimentional space.

4.  What is stochastic gradient descent (SGD)?

    - Is an ptimization algorithm used to minimize a loss function in machine learining, particularly in large-scale and online settings. Unlike traditional gradient descent, wich computers the gradient using the entire dataset, SGD approximates the gradient by calculating it for a randomly selected subset or a single data point (also known as a mini-branch or just one example) at each iteration. This approach makes SGD computationally faster and more suitable for large datasets, as it reduces the time per iteration. However the trade-off is that the voncergence path is noisier, and it might require more interations to reach a similar level of accuracy as traditional gradient descent.

5.  What is mini-batch gradient descent?

    - Is an optimization algorithm that combines the ideas of both Batch Gradient Descent and Stochastic Gradient Descent (SGD). Instead of using the entire dataset (Batch) or a single data point (SGD) to compute the gradient at each iteration, Mini-batch Gradient Descent calculates the gradient using a subset of the dataset, called a mini-batch.

    The mini-batch size is hyperparameter, typically ranging form 10 to few hundred examples. Mini-batch Gradient Descent offers a balance between the computational efficiency of SDG and the smoother convergence of Batch Gradient Descent. It leverages the benefits of vectorized operation on modern haedware while still being memory-efficient and providing a more stable convergence path taht pure SGD.

6.  What is a moving average?

    - A Moving Average is a static al technique used to smooth a time series data by calculationg the mean of a rolling window of observations over a perido of time. It's used to remove noise and isolate trends, making it easier to identify patterns and make forcasts.

    The formula for a simple moving average of window size n is:

    SMA[i] = (x[i] + x[i-1] + ... + x[i-n+1]) / n

    where x[i] is the i-th observation in the time series.

    To implement a moving average, you need to choose a window size n, and then slide the wisndow thwough the time series, updating the mean of the window at each step. One way to implement it is to use a for loop and update the sum and mean of the window using the precious value and the current calue of the time series. Another way is to using ht eprecious value and the current value of the time series. Another way is to use the convolution operation, which is a computationally efficient way to calculate moving averages using numpy or other numerical libraries.

7.  What is gradient descent with momentum? How do you implement it?

    - Gradient Descent with Momentum is a variation of the basic Gradient Descent algorithm used for optimization in machine learning models. It helps to speed up the learning process and reduce oscillations by incorporation a momentum term, which takes into account the previous gradients to calculate the new update. This momentum term acts as a low-pass filter, reducing the noise in the gradient updates, and helping the algorithm converge faster.

    The update rule of Gradient Descent with Momentum is as follows: 1. v(t) = β _ v(t-1) + (1 - β) _ ∇f(w(t-1)) 2. w(t) = w(t-1) - α \* v(t)

    w(t) is the current parameter value at iteration t.
    v(t) is the momentum term at iteration t.
    α is the learning rate.
    β is the momentum coefficient (a value between 0 and 1).
    ∇f(w(t-1)) is the gradient of the function f with respect to the parameters at iteration t-1.

8.  What is RMSProp?

    - Root Mean Square Propagation is an adaptive learning rate optimization algorithm designed for training neural netwoks. Is a improvement over the standard Gradient Descent and other adaptive learning rate methods like AdaGrad by addressing the problem of vanishing the expliding gradients during training.

    The main idea behind RMSProp is to maintain a moving avarage og the squared gradients and use this information to adjust the learning rate for each parameter indicudually. This results in faster convergence and better performance compared to other optimization techniques.

    The update rule for RMSProp is as follows:

    g(t) = ∇f(w(t-1))
    s(t) = γ _ s(t-1) + (1 - γ) _ g(t)^2
    w(t) = w(t-1) - α \* g(t) / √(s(t) + ε)
    Here,

    w(t) is the current parameter value at iteration t.
    g(t) is the gradient of the function f with respect to the parameters at iteration t.
    s(t) is the moving average of the squared gradients at iteration t.
    α is the learning rate.
    γ is the decay rate (a value between 0 and 1) for the moving average.
    ε is a small constant added for numerical stability, usually around 1e-8.

9.  What is Adam optimization?

    - Adam (Adaptive Moment Estimation) is an optimization algorithm for training neural networks. Adam has been widely adopted in deep learning due to its fast voncergence and roboustness to different hyperparameters and model architectures.

    The key idea behind Adam is to maintain separate moving avarages for both the gradiens (first moment) and the sequared gradients (second moment), and use these to adaptively adjuts the learning rate for each parameter individually.

    g(t) = ∇f(w(t-1))
    m(t) = β1 _ m(t-1) + (1 - β1) _ g(t)
    v(t) = β2 _ v(t-1) + (1 - β2) _ g(t)^2
    m_hat(t) = m(t) / (1 - β1^t)
    v_hat(t) = v(t) / (1 - β2^t)
    w(t) = w(t-1) - α \* m_hat(t) / (√v_hat(t) + ε)
    Here,

    w(t) is the current parameter value at iteration t.
    g(t) is the gradient of the function f with respect to the parameters at iteration t.
    m(t) is the first moment (moving average) of the gradients at iteration t.
    v(t) is the second moment (moving average) of the squared gradients at iteration t.
    m_hat(t) and v_hat(t) are bias-corrected versions of m(t) and v(t), respectively.
    α is the learning rate.
    β1 and β2 are the exponential decay rates for the first and second moment estimates, respectively (values between 0 and 1).
    ε is a small constant added for numerical stability, usually around 1e-8.

10. What is learning rate decay?

    - Learning rate decay, also known as learning rate scheduling or adaptive learnignt rate, is a technique used in training neural networks to adjust the learning rate during the optimization process. The main ideas to gradually decrease the learning rate as the training progresses. This helps the model to converge more effective by allowing it to make large updates in the beginning nad smaller, more precise updates as it gets closer to the optimal solution.

    Learning rate decay can improve the performance of the model and reduce the training time. The rationale behind this technique is that, as the model gets closer to the optimal solution, it requires finer adjustments to its parameters to avoid overshooting the minimum.

        There are several strategies for learning rate decay:

        1. Step decay: Reduce the learning rate by a factor after a fixed number of training epochs.
        2. Exponential decay: Decrease the learning rate at each step by a constant factor, leading to an exponential decline.
        3. Inverse time decay: Decrease the learning rate inversely proportional to the training step or epoch.
        4. Polynomial decay: Decrease the learning rate following a polynomial function of the training step or epoch.
        5. Cosine annealing: Decrease the learning rate following a cosine function of the training step or epoch.
        6. Cyclic learning rate: Oscillate the learning rate between lower and upper bounds, allowing it to increase and decrease periodically.
