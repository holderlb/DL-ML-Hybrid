# DL-ML-Hybrid

# Hybrid Approach

In this project, we introduce a hybrid DL-ML approach that uses a deep neural network for extracting features and a non-DL classifier to predict environmentally responsive transgenerational differential DNA methylated regions (DMRs), based on the extracted DL-based features. 

# Hybrid Approach

The main goal of this project is to predict whether a 1000bp DNA sequence is DMR or not. It use a DL network for extracting the features and an XGBoost model to build a classifier for predicting DMRs (or any other classification tasks). It can be used for any DNA classification problem with a minor modification in the shape of the input data, The main features of this project are: 

* Multi-class classification using deep learning and machine learning algorithm.
* interpretation of learned structures (motifs) in DNA sequences using a modified version of [pysster](https://github.com/budach/pysster) package.
* Train a hybrid model on  DNA sequences one-hot encoding representation. 
* Test a pre-trained model on a dataset (Input: one-hot encoding NumPy array of DNA sequences, labels)
* Train and test 5-fold cross-validation and report the classification results on the dataset. 
* Visualize the kernels of convolutional and dense layers.
* Hand-crafted features also can be added to the XGB by modifying the input argument of the `fit_ml_classifier` function. 

## Required packages

To use this classification the following packages are needed: 

* Python 3.x
* Tensorflow 1.x
* Keras 2.3.0
* Biopython 1.76
* Numpy 1.18.1
* Pandas 
* sklearn
* xgboost 0.90
* pysster 1.2.2

## Run the code

The parameter for the current version of the project is for [1000,5] NumPy one-hot encoding array.  If you want to train a hybrid model on a dataset you can load your Numpy array dataset instead of data and labels and modify the INPUT_SHAPE regarding the size of your sequence (e.g. If a sequence is 1000bp and the nucleotides in each region are A, C, T, G, and N the INPUT_SHPAE is [1000,5]). 

* You can use the following command for training a hybrid model: 

`python hybrid_model.py train-hybrid`

* You can use the following command to test a hybrid model on your dataset: 

`python hybrid_model.py test-hybrid`

* You can use the following command for a classification report of the hybrid method on a dataset using 5-fold cross-validation: 

`python hybrid_model.py 5-fold-hybrid`

* You can use the following command to visualize a layer of the DL network:

`python hybrid_model.py visualize`

* You can use the following command for training a DL model:
`python DL_model.py train-DL`

* You can use the following command to test a DL model on your dataset:
`python DL_model.py test-DL`

* You can use the following command for a classification report of a DL network  on a dataset using 5-fold cross-validation:
`python DL_model.py 5-fold-DL`
