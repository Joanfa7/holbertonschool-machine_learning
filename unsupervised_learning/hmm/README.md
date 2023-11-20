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
6. What is a regular Markov chain?
7. How to determine if a transition matrix is regular
8. What is an absorbing state?
9. What is a transient state?
10. What is a recurrent state?
11. What is an absorbing Markov chain?
12. What is a Hidden Markov Model?
13. What is a hidden state?
14. What is an observation?
15. What is an emission probability/matrix?
16. What is a Trellis diagram?
17. What is the Forward algorithm and how do you implement it?
18. What is decoding?
19. What is the Viterbi algorithm and how do you implement it?
20. What is the Forward-Backward algorithm and how do you implement it?
21. What is the Baum-Welch algorithm and how do you implement it?