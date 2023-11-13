# Clustering

## Learning Objectives

1. What is a multimodal distribution?
    A multimoda distribution in statistics is a probability distribution with more than one peak, or "mode". Each mode corresponds to a local maximum in the probability density function. In the context of clustering in ML, this is significant because it suggests the presence of multiple subgroups within the data.

    For example, consider a dataset representing teh heights of a mixed group of adults and children. the distribution of heights will likely be multimodal, with one peak corresponding to the average height of children and another peak representing the average height of adults. In clustering, identifying these modes helps in understanding the inherent groupings in the data.

2. What is a cluster?
    A cluster in teh context of ML and statistics refers to a group of data points or objects that are similar to each other within the same group and dissimilar to the objects in other groups. Clustering is the process of diving a set of data points into these groups based on their characteristics.

    For instance, in a dataset of customers' shopping behaviors, a cluster might consist of customers who buy similar items, shop at similar times, or have similar spending habits. The goal of clustering algorithms, like K-means, hierarchical clustering, of DBSCAN, is to identify these naturally occurring groupings in the data, other without prior knowledge of the group identities (this is known as unsupervised learning). Clustering is widely used for market segmentation, organizing large sts of data, pattern recognition, and anomaly detection.

3. What is cluster analysis?
    Cluster analysis, also known as clustering, is a technique in ML and statistics used to group of set of objects in such a way that objects in the same group (called a cluster) are more similar to each other than to those in other groups (clusters). It's a method of unsupervised learning, meaning it's used to draw inferences from datasets consisting of input data without labeled responses. 

    The goal of cluster analysis is to discover underlying patterns in data. It involves various algorithms and methods to group objects of similar kinds into respective categories. Common clustering techniques include:

        - K-Means Clustering
        - Hierarchical Clustering
        - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    
    Applications of cluster analysis are widespread and include market research, pattern recognition, data analysis, and image processing. For example, in market research, cluster analysis can help identify distinct groups of customers based on their purchasing behavior, demographics, and preferences, enabling more targeted marketing strategies.

4. What is “soft” vs “hard” clustering?
5. What is K-means clustering?
6. What are mixture models?
7. What is a Gaussian Mixture Model (GMM)?
8. What is the Expectation-Maximization (EM) algorithm?
9. How to implement the EM algorithm for GMMs
10. What is cluster variance?
11. What is the mountain/elbow method?
12. What is the Bayesian Information Criterion?
13. How to determine the correct number of clusters
14. What is Hierarchical clustering?
15. What is Agglomerative clustering?
16. What is Ward’s method?
17. What is Cophenetic distance?
18. What is scikit-learn?
19. What is scipy?