# Hidden Markov Models

## Learning Objectives

1. What is the Markov property?

    The Markov property, also known as the "memoryless property", is a fundamental concept in probability theory and forms the basis of Markov Models, including Hidden Markov Models.

    Definition:
        The Markov property states that the future stat of a process depends only on the current state and not on the sequence of events that preceded it. In other words, it has no "memory" of the past.

    The Markov property's focus on the present state, disregarding the path taken to reach that state, makes it powerful and simplifying feature in modeling various stochastic(random) processes. It's particularly useful in areas like economics, game theory, and any field dealing with predictive models where the future state is a probabilistic function of only the current state.

2. What is a Markov chain?

    A Markov chain is a mathematical model that describes a sequence of possible events in which the probability of each event depends only on the state attained in the previous event. This concept is a direct application of the Markov property, which implies that the future state of a process depends only on the current state, not on the sequence of events that preceded it.

    Markov chains are a versatile tool in statistical modeling, offering a way to model random processes across various fields. Their simplicity lies in the memoryless property, where predictions are based solely on the current state, making them particularly useful in situations where historical data in less relevant or too complex to include in the model.

3. What is a state?

    These are the distinct statues that the process cab be in. For example, in weather modeling, the states could be "sunny" and "rainy".

4. What is a transition probability/matrix?

    A transition is the process moves form one state to another, and these movements are called transitions.

    Each transition has a probability associated with it, determining how likely it is to move from one state to another.

5. What is a stationary state?
    Is a condition where the probabilities associate with the different states in the system remain constant over time. this concept is crucial for understanding long-term behavior in probabilistic models.

6. What is a regular Markov chain?

    A regular Markov chain is a specific type of Markov chain characterized by a property that guarantees convergence to a unique stationary distribution, regardless of the initial state. This type of Markov chain plays a crucial role in understanding the long-term behavior of stochastic processes.

7. How to determine if a transition matrix is regular

    Determining the regularity of a transition matrix is a matter of checking whether there is a certain number of steps after which it's possible to get form any state to any other state with a non-zero probability. This feature is pivotal in understanding the long-term behavior of Markov chains, especially in modeling scenarios where the system is expected too reach a steady state.



8. What is an absorbing state?

    Absorbing states in a Markov chain is a state that, once entered, cannot be left. In other words, once the process reaches an absorbing state, it remains there indefinitely.

    Characteristics:
        1. No Exit
        2. Self-loop Probability
        3. Permanent State

9. What is a transient state?

    Transient state are an integral part of the structure of Markov chains and are key to understanding the dynamics of stochastic process. The  represent temporary phases through which the process passes before potentially reaching more stable, recurrent states. Identifying and analyzing these states can provide significant insights about the behavior and future evaluation of complex stems modeled by Markov chains.

    Characteristics:
        1. Non-Permanent
        2. Finite Number of Visits
        3. Probability of Exit

10. What is a recurrent state?

    A recurrent state in Markov chain is a state that, once visited, has a probability of one of being visited again infinitely often. This is in contrast to a transient state, where the probability of returning to the state infinitely often is less than one.

    Characteristics:
        1. Repeated Visits
        2. Probability of Return
        3. Long-Term Behavior

11. What is an absorbing Markov chain?

    An Absorbing Markov chain is a special type of Chain that contains at least one absorbing state. An absorbing state, as previously mentioned, is one that once entered, cannot be left. The presence of such a state in Markov chain has a significant implications for the behavior of the chain over time.

    Characteristics:
        1. Presence of Absorbing States
        2. Possibility of Absorption
        3. Long-Term Behavior

12. What is a Hidden Markov Model?

    A Hidden Markov Model (HMM) is a statistical model used to describe a system that is a Markov process with unobservable (hidden) states. HHM are particularly known for their applications in temporal pattern recognition such as speech, handwriting, gesture recognition, part-of-speech tagging, and bioinformatics.

13. What is a hidden state?

    The states of the model are not directly visible. Each hidden state is associate with a probability distribution over observable events.

14. What is an observation?

    These are the events or data points that are visible or measurable. They are a result of the process being in one or the hidden states.

15. What is an emission probability/matrix?

    These are the probabilities of transitioning from one hidden state to another. The represent the Markov chain aspect of the model.

16. What is a Trellis diagram?

    A trellis diagram is a graphical representation used primarily in the field of coding theory and digital communications, as well as in HMM. It provides a visual way to represent the state transitions in a system over time, particularly useful in the context of decoding and analyzing sequential data.

    They are used to visualize the state transitions and possible paths in HMMs, aiding in understanding the underlying process and in algorithms like Viterbi algorithm for finding the most likely sequence of hidden states.

17. What is the Forward algorithm and how do you implement it?

    Is a fundamental procedure used in HMM to compute the probability of observing a particular sequence of observation given the model parameters. It's essential for many tasks involving HMMs, such as evaluating model fit and predicting state sequences.

    Key Steps:
        1. Initialization
        2. Recursion
        3. Termination

18. What is decoding?

    Determining the most likely sequence of hidden states given the sequence of observations, often using the Viterbi Algorithm.

19. What is the Viterbi algorithm and how do you implement it?

    The Viterbi algorithm is a dynamic programming algorithm used in HMM to find the most probable sequence of hidden states (also known as the Viterbi path) given a sequence of observations. This is particularly usual in applications like speech recognition, where we want to know the most likely sequence of words (hidden states) that produced a given sequence of sounds (observations)

    Key Steps:
        1. Initialization
        2. Iteration
        3. Termination
        4. Path Tracing

20. What is the Forward-Backward algorithm and how do you implement it?

    The Forward-Backward algorithm is a fundamental technique used with HMM to compute the posterior probabilities of the hidden states given a sequence of observations. It combines two procedures: the Forward algorithm, which computes the probability of the sequence up to a certain point, and the Backward algorithm, which computes the probability from a certain point to the end of the sequence.

    Overview:
        1. Forward Pass: Calculates teh probability of each state at each time step, given the observed sequence up to that point.

        2. Backward Pass: Calculates the probability of the observed sequence from a certain point in time the end, given each state.

        3. Combining Forward and Backward Probabilities: The posterior probability of each state at each time step is computed by combining the forward and backward probabilities.


21. What is the Baum-Welch algorithm and how do you implement it?

    The Baum-Welch algorithm is a type of Expectation-Maximization (EM) algorithm used to find the unknown parameters of a HMM. It's particularly useful when you have a sequence of observed data and want to estimate the HMM parameters (transition probabilities, emission probabilities, and initial state probabilities) that most likely produced this sequence.

    Key Steps:
        1. Initialization: Start with initial guesses for the model parameters.

        2. Iterative Process: Repeat the following until convergence.

            1. E-step: Compute the expected occurrence of the transitions and emissions using the
            Forward-Backward algorithm.

            2. M-step: Update the model parameters based on these expected occurrences.
