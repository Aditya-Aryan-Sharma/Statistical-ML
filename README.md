# SML

I implemented all those new Algorithms that I learn in Statistical Machine Learning.

In A1_2020015.py:
I visualized gaussian random variable geometrical representation and tried to find an optimal decision boundary given two random variables.

In A2_2020015.py :
I generated 200 samples for binary classification, used first 100 samples as testing samples and computed MLE and Linear Discriminant Analysis and then used it to 
classify my test samples with accuracy of more than 80%.
I also tried to compute Principal Component Analysis from scratch for a 2x2 arbitrary matrix.

In LDA_cifar10.py:
I loaded custom cifar-10 dataset and visualized 5 samples of each class in the form of images from the training set. Then I applied Linear Discriminant Analysis on the 
training set to obtain a classifier which then I used to predict test samples with more than 80% accuracy.

In PCA_MNIST.py:
I tried to visualize the most efficient dimension reduction using PCA. In this I performed dimensionality reduction paradigm on MNIST dataset using deep learning modules. Then I performed LDA on transformed data in order to obtain a classifier which is used later to classify testing samples.
