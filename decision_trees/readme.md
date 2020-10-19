# Decision Trees

Decision trees use tree-like models of decisions and their possible consequences. It can be used to easily represent algorithms with only conditional statements(no loops). In machine learning, they are used mostly for classification. 

Here is an example of a decision tree for classification. The objective is to find if a person is fit or not. The tree can be very easily interpreted, which is one of the positives of decision trees (used for analysing structures in data).

![Decision tree example](../Documents/Images/decision-tree-3.png)

Our objective is to have this tree being built by analysing a dataset. For this, let us try to analyse this tree and why it works.

## Understanding the fitting process

If we look at the above tree, the top decision, Age < 30 divides the dataset into two (age<30 and otherwise). Each of these are easier to further classify than the source dataset. Our objective is to find the best parameter to divide the datasets into such halves. Here, for the root node, it was Age, and the threshold was 30. In order to do this, we use a score, called the **Gini impurity**.

It is a measure of how uniform the dataset is, with respect to the outputs desired. For example, a dataset with only one class will have a gini impurity of 0 (perfectly pure), while one with a uniform distribution of 3 classes will have a higher gini impurity. It is defined as:

![Gini impurity equation](../Documents/Images/decision-tree-gini.png)

Where p(i) is the proportion of class i, in the full dataset. FOr example, if there is 50 elements of class 1 in a dataset with 150 elements, ```p(1) = 1/3```. For a dataset with 2 classes, both uniformly distributed p(1)=p(2)=0.5, the gini impurity will be computed as

![Gini impurity example](../Documents/Images/decision-tree-gini-example.png)

For some decision node, this is computed for both the splitted datasets, and their weighted sum (wieght is total number of elements in each of these) is found. The best way to split the dataset is found by trial and error by checking all the parameters, with different  thresholds each and computing the gini impurity of their splits. This is then repeated for the next level, using the splitted dataset.

If the gini impurity after splitting does not improve from the original dataset, them the decision node is deemed useless. In such a case the node is made into a 'leaf node'. That is, upon reaching that node, the sample is classified as some output. While fitting, the mode (most frequently occuring class in that dataset) is chosen as the output. Leaf nodes can also be formed by setting a maximum depth to the tree.

![Decision tree ](../Documents/Images/decision-tree.png)

## References

1. [Geeksforgeeks](https://www.geeksforgeeks.org/decision-tree-introduction-example/#:~:text=Decision%20tree%20uses%20the%20tree,attributes%20using%20the%20decision%20tree.)

2. [Statquest youtube channel](https://www.youtube.com/watch?v=7VeUPuFGJHk)

3. [Medium article by Artyom Kulakov](https://medium.com/datadriveninvestor/easy-implementation-of-decision-tree-with-python-numpy-9ec64f05f8ae)

4. [Towardsai.net](https://towardsai.net/p/programming/decision-trees-explained-with-a-practical-example-fe47872d3b53)

5. [Gini impurity - victorzhou](https://victorzhou.com/blog/gini-impurity/)