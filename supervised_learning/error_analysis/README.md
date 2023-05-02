# Error Analysis

1. What is the confusion matrix?

   - A confusion matrix is a table that is used to describe the perfomrance of a classification algorithm, typically a supervised learning technique, It displays the number of true positive (TP), true negative (TN), flase positive (FP), and false negative (FN), and flase negative (FN) predictions for each class, helping to visalize the accuracy, precision, recal, and other perfomrance metrics of the model.

2. What is type I error? type II?

   - Type I error (also called false positive) occurs when the classifier incorrectly labels a negative instance as positive. Type II (also call a false negative) occurs when the classifier incorrectly labels a positive instance as negative.

3. What is sensitivity? specificity? precision? recall?

   - Sensitivity (also known as recall or true positive rate) is the proportion of actual positive instance that classifier correctly identifies as positive. Specificity (also knon as true negative rate) is the proportion of actual negative instance that the classifier correctly identifies as negative. Precision (also known as positive predictive values) is the proposrtioin of true positive instances among the intances that the classifier identifies as positive. Recall is the same as sensitivity.

4. What is an F1 score?

   - The F1 score is a harmonic mean of precision and recall, providing a single value to evaluate the trade-off between these two metrics. it ranges between 0 and 1, with 1 being the best possible value, and it is particularly useful when dealing with imbalanced datasets.

5. What is bias? variance?

   - Bias is the error introduced by approximating a real-world problem with a simplified model. It measures how far of the model's predictions are from the correct vlaues on average. Variance is the error introduced by model's sensiticity to small fluctuation int he trainig data. It measures how much the model's predictions cary bewtwee different training sets.

6. What is irreducible error?

   - Is the error that cannot be eliminated by improving the model. It is caused by inhernet noise or randomness in the data and represents the lowest possible error rate for a given problem.

7. What is Bayes error?

   - Bayes error is the minimum possible error rate fro a classification problem, achived by the optional classifier that knows the true underlying probability distribution of the data. In practice, Bayes error not atteinable since the true distribution is usually unknown.\

8. How can you approximate Bayes error?

   - You can approximate Bayes error by using techniques such as cross-validation with multiple models or ensemble learning to estimate the best possible perfomrance on a given dataset. Alternaaticely, you can use domain knowledge or theoretical analysis to estimate the underlying distribution of the data.

9. How to calculate bias and variance

   - To calculate bias and variance, you can use techniques such as bootstrapping or k-fold cross-validation to create multiple training sets and test set. Compute the average prediction error across thest sets for bias, and the variance of the prediction error for variance.

10. How to create a confusion matrix
    - To create a confusion matrix, first, train your classifier on a training dataset and make predictions on a test dataset. Then, count the occurrences of true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions for each class. Arrange these values in a matrix format with rows representing the actual classes and columns representing the predicted classes.
