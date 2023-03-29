# 0x02. Calculus

## Learning Objectives

1. What is Summaiton and Product Notation?

   - Summation notation is a concise wayof representin the sum of a colleciton of numebrs. It involves writing the expression to be summed using the Greek capital letter sigma(Σ), where the variable of summation is written below the sigma, and the limits of summatino are written above and below the sigma.

   - Product notations is similar to sumaitn notation , but instead represents the product of a collection of numebrs, It involves writing the expression to be multiplied using the Greek capital letter pi (Π), where the variable of multiplication is written below the pi, and the limits of multiplication are written above and bellow the pi.

[Click for Example: Notation](https://scontent-mia3-1.xx.fbcdn.net/v/t1.6435-9/106803074_140803567650264_303350664615146918_n.jpg?_nc_cat=111&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=8D_N0y3xyucAX-Fp7Tc&_nc_ht=scontent-mia3-1.xx&oh=00_AfABa5BSMt7ktyzEyIGAC0GSj6uyOGVJU0E9IbKbtDnWBA&oe=644AB93D)

2. What is a series?

   - A series is the **sum** of an infinite sequence of terms. A series can be represented using the summatino notation, where the variable of summatino takes on all integer values tarting from some lower limit.

3. What are the must common series used in Machine Leanring?

   - Arithmetic series: This is a series in which the difference between consecutive terms is constant. In Machine Learning, arithmetic series are commonly used for time series forcasting andtrend analysis.

   - Gometric Series: This series s a series in which each term is a power of a fixed base. In machine Learning, geometric series are commonly used for exponential growht or decay models.

   - Power Series: This is a series in which each term is a power of a fixed base. In Machine Learning, power series are commonly used for polynomial regression models.

   - Fouries series: This is a series that represents a preiodic function as a sum of a sine and a cosine funcitons. In Machine Learning, Fourier series are commonly used for signal processing and image analysis.

   - Taylor series: This is a series that represents a functino as an infinite sum of it derivatives evaluated at a signal point. In Machine Learning, Taylo series are commonly uese for function aproxumation and optimization.

4. What is a derivative?

   In calculus, the derivative of a function represents the rate of change of the function with respect to its input. More specifically, the derivative gives the slope of the tangent line to the function at a given point.

   In ML, derivatives are commonly used for optimization. The goal of optimization in machine learning is to find the parameters of a model that minimize a certain loss function, which measueres the difference bewteen the predicted output of the model and the actual output.

   Common optimization algorithms in machine learning, such as gradient descent and its variations, rely heavely on the calculation of derivatives. These algorithms use the derivative of the loss function with respect to the model parameters to update the parameters in each iteration of the optimization process. Therefore, understanding derivatives and their application is crucal for building and training effective machine learning models.

5. What is the product rule and how is applied to Machine Learning?

   The porudct rule is a funcitonal rule of differentional in calculus that allows you to find the derivative of porudct of two funcitons. Specifically, if you have two funciton f(x) and g(x), the product rule states that the derivative of their product, denoted as (f(x) \* g(x)), is given by:

   (f(x) _ g(x))' = f'(x) _ g(x) + f(x) \* g'(x)

   In the context of Machine Learning, the product rule is used when calculating the gradiants of functions that are composed of multiple sub-functins. For example, in a neural network, each layes is typically composed of multiple sub-funcitons(suchas activation funcitnos and linear transformations), and the gradient of the output of the layer with respect to its input can be calculated using the product rule.

6. What is tha chain rule?
   The cain rule is another fundamental rule of differentiation in calculus that allows you to find the derivatice of a composition of two funcitons. Specifically, if you have two functions f(x) and g(x), where g(x) is a function of f(x), the chain rule states that the derivatice of their composition, denoted as g(f(x)), is given by:

   (g(f(x)))' = g'(f(x)) \* f'(x)

   In other words, you first find the derivative of the outpu function (g) with respect to its input (f(x)), and then mulply it by the derivative of the inner funciton (f) wiht repsect to its input (x).

   In the contex of ML, the chain rule is used extensively in backpropagation, which is the algorithm used to train neural networks. In backpropagation, the chain rule is used to calculate the gradiants of the loss function with respect to the weights of the network. Specifically, the gradient of the loss function with repsect ot the weith of a particular layer can be calculated by chaining toheter the graients of the loss function with respect to the outputs of the layer of the gradiants of the outputs wiht repsect to the inputs, using the chain rule.
