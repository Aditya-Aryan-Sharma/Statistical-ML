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
custom cifar-10 dataset:https://drive.google.com/file/d/1k99CJ9XCtK6zM7b0XthRYmcRc2W5B4wJ/view

In PCA_MNIST.py:
I tried to visualize the most efficient dimension reduction using PCA on MNIST dataset. In this I performed dimensionality reduction paradigm on MNIST dataset using deep
learning modules. Then I performed LDA on transformed data in order to obtain a classifier which is used later to classify testing samples.
custom MNIST dataset:https://drive.google.com/file/d/1wguGAR2HRZ9WZZ91EKrpv3onD4vLHm5c/view

In FDA_FMINST.py:
I computed Fisher's Discriminant Analysis from scratch on Fashion-MNIST dataset. Then used the transformed data to compute LDA and obtained a classifier which is used
later to classify testing samples.Then I tried to predict overall accuracy and class-wise accuracy.
custom Fashion MNIST dataset:https://drive.google.com/file/d/1tBKDbHlkzy1K2JFMF6E59DQ5wGCUMQdY/view

In PCA_FDA_LDA.py:
I used PCA for dimensionality reduction from 784x50000 to 15x50000 and then used FDA to help in classification of test samples.Then I applied LDA for obtaining a 
classfier and tried to classify test samples.
