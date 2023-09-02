# Neuroapoptosis Model with a CNN Classifier

## Description
This project examines the effects of neuroapoptosis on the ability for an artificial perceptual model (a CNN classifier) to predict the classification of an image. It constructs a VGG16 CNN model for classifiying bodies vs. nonbodies (https://academic.oup.com/cercor/article/29/1/215/4653734). Then, the program removes proportions of neurons randomly from the last convolutional layer, and uses the new model to examine the accuracy of the model. 

This program used Tensorflow and Keras (for constructing the model) and Sklearn (for applying PCA). 

First, the program extracts the data (images), assigns them labels, and splits them into a training set, a validation set, and a test set, in a 80-10-10 ratio, approximately. 

Next, a model is constructed with a VGG16 input to extract image components and features, then applies PCA to extract the principal components. Finally, an SVM accepts those principal components as input to output a binary classification.

The model is then trained on the training and validation set, and hyperparameters of the model (learning rate and the L2 hyperparameter) are optimized here.

The program then cycles through various removal rates, removing that proportion of neurons from the last convolutional layer of the VGG16 network, and passes images through to evaluate the accuracy of the model.

The experiment found that the model remained relatively stable at a 90% accuracy until reaching an 80% removal rate. At higher removal rates, the model's average accuracy fell sharply.

![neuroapoptosis_accuracies_august](https://github.com/adityamkk/neuroapoptosis/assets/73001560/4027b688-e159-4a03-8ccd-8549b3446261)

In the future, this program may implement various other metrics of analysis, such as precision and recall instead of accuracy.

## Installation
Download the "NeuroapoptosisModel.ipynb" file and upload it to Google Drive. Simply open Colab and run.

## Sources

https://www.tensorflow.org/guide/keras/train_and_evaluate

https://www.learndatasci.com/tutorials/hands-on-transfer-learning-keras/

https://www.learndatasci.com/tutorials/convolutional-neural-networks-image-classification/#UsingImageDataGeneratorfortraining

https://towardsdatascience.com/transfer-learning-with-vgg16-and-keras-50ea161580b4

https://github.com/sudhir2016/Google-Colab-3/blob/master/VGG16.ipynb

https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53

https://stackoverflow.com/questions/66571756/how-can-you-get-the-activations-of-the-neurons-during-an-inference-in-tensorflow

https://gist.github.com/martinsbruveris/1ce43d4fe36f40e29e1f69fd036f1626
