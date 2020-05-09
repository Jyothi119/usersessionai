# usersessionai

1.Firstly I was look into the task and analyzed is it a classification or regression algorithm.In the given task I have to predict weather the user is going to the advertisement website or not this itself is indicating that it is a classification algorithm.
2.Then I started coding by importing all the useful libraries and dataset.
3.After that I performed the data cleaning operations, I mean to see any null values or blanks are present in dataset.
4.Then I checked whether my data is imbalanced or balanced by plotting.
5.Then I find the correlation between the variables this means that it tells how one variable is depending on the other. In the given dataset no two variables are closely related.
6.In the above step I have found that my dataset has imbalancing problem to eliminate this i performed the sampling to equal the both positive and negative points. The problem with the imbalanced data is the output is going to be biased for the majority of the datapoints. I will explain with the example suppose here in my dataset has 1000 datapoints but out of 1000 datapoints 900 is positive and 100 is -ve so my model will be trained by majority of the datapoints like some 650. I may get the accuracy of 90% but this is not the best metric when compared with the imbalanced data it is going to mislead our model and give wrong predictions however the accuracy is good. When come to imbalancing data set there are other better metrics like confusion matrix, precision, F1 Score and recall. I have determined the confusion matrix for my problem here it will give the table of correct predictions and incorrect predictions.
7.First, I have to convert the imbalanced dataset to balanced dataset to avoid the biasing output by using sampling techniques and then perform the metrics.
8.Eventhough the accuracy got decreased after performing sampling but the performance metrics was good.
9.Then I normalized the data by using standard scalar method in python to normalize any big variances.
10.After this I performed the exploratory data analysis to find the patterns. The final thing is training the model by using training data and then find the confusion matrix to describe the performance of my classification model.
using matthew's corrleation coefficient i predicted the quality of my model using the values obtained from the confusion matrix.
