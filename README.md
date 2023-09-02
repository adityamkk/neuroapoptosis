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

## Experiment and Analysis Plan


1. Create a VGG16 image classification model
   1. Solving Bodies vs. Non-bodies (https://academic.oup.com/cercor/article/29/1/215/4653734)
   2. Give images two distinct labels → bodies and non-bodies
   3. Split the dataset into 83% training, 7% validation, and 10% testing (K. K. Dobbin and R. M. Simon, Optimally splitting cases for training and testing high dimensional classifiers, BMC Med. Genet. 4 (2011), no. 1, 1– 8.   https://bmcmedgenomics.biomedcentral.com/articles/10.1186/1755-8794-4-31?report=reader )
   4. Resize images into a 224x224 format since VGG16 only accepts that size
   5. Instantiate a new VGG16 model
      - Weights are based from imagenet
      - Remove the top and replace with input stream
      - VGG model specifically is not trainable
   6. Attach 3 Layers to the end of the VGG16 model
      - One Flatten Layer → in order to reduce the dimensions of the output of the VGG16 model → make output 1D
      - Apply PCA to the output of the Flatten layer
         1. 62 categories → 95% of variance explained by 62 categories http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/
         2. svd=randomized
      - One SVM layer → Make final classification
         1. Activation is linear
         2. Has a L2 Ridge regression regularizer → to make sure weights don’t become zero (Regularization in Machine Learning || Simplilearn.)
         3. Regularizer hyperparameter is optimized through keras-tuner Hyperband
         4. Hyperparameter is between 0.0 and 0.1 [NEED SOURCE]
   7. Compile Model
      - Optimizer is adam (standard)
         1. Learning rate is set to 0.01 instead of the default (0.001)
         2. This learning rate was found through repeated observation of different learning rates. The default rate was too low for the model to adjust weights quickly enough.
      - Loss function is hinge (for the SVM)
      - Only evaluates accuracy (precision may be more useful)
      - 50 epochs maximum and batch size of 20 allows for 100 iterations → almost the number of images in the dataset.
2. Find BSI (Body Selectivity Index) for each neuron in penultimate layer
   1. Split testing dataset into bodies and non-bodies (since BSI requires two firing rates)
   2. Note: Firing rate of neuron is equal to its activation
   3. For both categories, feed in single stimulus, and record activations of a neuron in layer
   4. Average the activations of the neuron for all stimuli for both categories separately
   5. Subtract both mean body and mean non-body activations by the baseline activation to find R_B and R_NB.
   6. Calculate the BSI for each neuron.
3. Experimental Procedure
   1. Threshold all neurons with |BSI| >= 0.33
   2. Repeat 100 times for each deletion within layer:  0%-100% removal rates, 10% skips
      - Delete randomly selected (use choice) neurons
      - Record proportion of neurons above thresholded BSI
      - Test model for accuracy and record
      - Place results into array
   3. Plot results for each deletion percentage on scatterplot
   4. Intra-deletion analysis
      - Find mean and standard deviation for accuracies
      - Select BSI threshold rates where corresponding accuracies are more than 1 stdev away from the mean
      - Find the average threshold rate and compare to the rates found in 3dii
      - Find statistically significant difference between threshold rates → this can determine if selectivity plays an important role for classification → does removing a selective neuron impact classification more than a non-selective neuron?
   5. Inter-deletion analysis
      - Statistically significant difference between mean accuracies? Two-prop z-test


## Sources

https://www.tensorflow.org/guide/keras/train_and_evaluate

https://www.learndatasci.com/tutorials/hands-on-transfer-learning-keras/

https://www.learndatasci.com/tutorials/convolutional-neural-networks-image-classification/#UsingImageDataGeneratorfortraining

https://towardsdatascience.com/transfer-learning-with-vgg16-and-keras-50ea161580b4

https://github.com/sudhir2016/Google-Colab-3/blob/master/VGG16.ipynb

https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53

https://stackoverflow.com/questions/66571756/how-can-you-get-the-activations-of-the-neurons-during-an-inference-in-tensorflow

https://gist.github.com/martinsbruveris/1ce43d4fe36f40e29e1f69fd036f1626
