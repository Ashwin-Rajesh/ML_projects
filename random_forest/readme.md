# Random Forest

Random forest is an **ensemble** learning method, which means it combines various individual machine learning models to result in greater overall performance. The method used is called **Bagging**.

Basically, a subset of the initial dataset is created, consisting of random features and random rows. This subset is used to 'grow' individual trees. This is where the method gets its name. Since all these trees are randomized, they learn to classify data using different methods. The idea is that by taking the average performance of these trees, we can achieve better **generalisation** and also better performance than individual trees.

## Implementation

I have used the same decision tree class here as well, to create a random forest class that stores root nodes of multiple trees in the forest. The API is similar to the decision tree API.

