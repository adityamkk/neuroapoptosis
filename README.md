# Neuroapoptosis Model with a CNN Classifier

## Description
This project examines the effects of neuroapoptosis on the ability for an artificial perceptual model (a CNN classifier) to predict the classification of an image. It constructs a VGG16 CNN model for classifiying bodies vs. nonbodies (https://academic.oup.com/cercor/article/29/1/215/4653734). Then, the program removes proportions of neurons randomly from the last convolutional layer, and uses the new model to examine the accuracy of the model. 

This program used Tensorflow and Keras (for constructing the model) and Sklearn (for applying PCA). 

In the future, this program may implement various other metrics of analysis, such as precision and recall instead of accuracy.

## Installation
Download the "NeuroapoptosisModel.ipynb" file and upload it to Google Drive. Simply open Colab and run.
