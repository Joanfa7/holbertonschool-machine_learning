# Probability

<img src="./images/probability-formula-cheat-sheet-1.png" alt="Alt text" title="Probability image">

## Learning Objective

1. What is probability?

   - Probability is a measure of the likelihood that a particular event or outcome will occur, expressed as a value between 0 and 1. In ML, probability helps in making predictions, estimating uncertainties, and modeling real-world processes throught techniques like classification, regression, and Bayesian infrerence.

2. Basic probability notation

   1. P(A): Probability of event A occurring.
   2. P(A ∩ B): Proability of both events A and B occurring, also known as the intersection.
   3. P(A U B): Probability of either event A or event B or both occurring, aldo known as the union.
   4. P(A | B): Conditional probability, meaning the probability of event A occuring given the event B has occurred.
   5. P(Ac): Probability of the complement of event A meaning the probability of A not occuring.
      In ML, these notations help in understanding and calculating probabilities to make predictions, train models and evaluate their performance.

3. What is independence? What is disjoint?

   - Independence: Two events, A and B, are independent if the occurrence of one event does not influence the probability of the other event. Mathematically, events A and B are independent if P(A ∩ B) = P(A) \* P(B). In ML, independence is often assumed between features in certain models to simplify calculations and reduce computational complexity.

   - Disjoint: Two events, A and B, are disjoint (or mutually exclusive) if the cannot occur at the same time, meaning their intersection is empty. Mathematically, events A and B are disjoint if P(A ∩ B) = 0. Disjoint events are relevant in ML when working with classification tasks, where and observation belongs to only one class, and the classes are mutually exlusive.

4. What is a union? intersection?

   - The union of two events, A and B, represents the event that either A, B or both occur. In terms of probability, the union is calculated as P(A ∪ B) = P(A) + P(B) - P(A ∩ B). In ML, union can be useful when combining probabilities of multiple events, such as in esemble methods that combine preodictions form different models.

   - Intersection: The intersection of two events, A and B, denoted as A ∩ B, represents the event where both A and B occure simultaneously. In terms of probability, the intersection is claculated as P(A ∩ B). In ML intersections van be useful when evaluating dependencies between events, such as Bayesian nerworks or hidden Markov model, where joint probabilities are required.

5. What are the general addition and multiplication rules?

   - The general additional rule is used to find the probability of the uniot of two events A and B. I states:

     P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

     This rule takes into account the posibility of A nad B overlapping, so it subrtacts the interseciton to avoid double-counting. In ML, the general rule is useful when combining probabilities of multiple events or predictions, such as in enseble methos or multi-class classificaiton problems.

6. What is a probability distribution?

7. What is a probability distribution function? probability mass function?

8. What is a cumulative distribution function?

9. What is a percentile?

10. What is mean, standard deviation, and variance?

11. Common probability distributions
