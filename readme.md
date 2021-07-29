# Hybrid Approach

In this project, we introduce a hybrid DL-ML approach that uses a deep neural network for extracting features and a non-DL classifier to predict environmentally responsive transgenerational differential DNA methylated regions (DMRs), based on the extracted DL-based features. 

# Hybrid Approach

The goal is to build a classification model that takes a region of the genome as input and predicts the region’s susceptibility to develop an environmentally induced transgenerational alteration in differential DNA methylation regions (DMRs) in the F3 generation from an ancestrally exposed F0 generation (great grandmother). 

It uses a DL network for extracting the features and an XGBoost model to build a classifier for predicting DMRs (or any other classification tasks). It can be used for any DNA classification problem with a minor modification in the shape of the input data, The main features of this project are: 

* Multi-class classification using deep learning and machine learning algorithm.
* interpretation of learned structures (motifs) in DNA sequences using a modified version of [pysster](https://github.com/budach/pysster) package.
* Train a hybrid model on  DNA sequences one-hot encoding representation. 
* Test a pre-trained model on a dataset (Input: one-hot encoding NumPy array of DNA sequences, labels)
* Train and test 5-fold cross-validation and report the classification results on the dataset. 
* Visualize the kernels of convolutional and dense layers.
* Hand-crafted features also can be added to the XGB by modifying the input argument of the `fit_ml_classifier` function. 

## Network Architecture 
The method takes a 1000bp region of the DNA sequence as input and produces a classification for the region as to whether it will be susceptible to environmental exposure as evidenced by differential methylation. The proposed hybrid model is consists of a deep learning (DL) network that is trained using the dataset and a traditional machine learning (ML) classifier that is also trained using the dataset, but with the input region re-expressed using features extracted from a layer of the deep learning network. 

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



## Data Availability 

The Skinner laboratory at Washington State University has produced several datasets based on the rat genome that identify DMRs in the F3 generation male sperm after exposing the F0 generation to one of nine toxicants: atrazine, dichloro-diphenyl-trichloroethane (DDT), glyphosate, vinclozolin, pesticides, dioxin, jet fuel, methoxychlor, and plastics.

All molecular data has been deposited into the public database at NCBI under GEO #s: [GSE113785](https://www.ncbi.nlm.nih.gov/search/all/?term=GSE113785) (vinclozolin), [GSE114032](https://www.ncbi.nlm.nih.gov/search/all/?term=GSE114032) (DDT), [GSE98683](https://www.ncbi.nlm.nih.gov/search/all/?term=GSE98683) (atrazine), [GSE155922](https://www.ncbi.nlm.nih.gov/search/all/?term=GSE155922) (jet fuel), [GSE157539](https://www.ncbi.nlm.nih.gov/search/all/?term=GSE157539) (dioxin), [GSE158254](https://www.ncbi.nlm.nih.gov/search/all/?term=GSE158254) (pesticides), [GSE158086](https://www.ncbi.nlm.nih.gov/search/all/?term=GSE158086) (methoxychlor), [GSE163412](https://www.ncbi.nlm.nih.gov/search/all/?term=GSE163412) (plastics), and [GSE152678](https://www.ncbi.nlm.nih.gov/search/all/?term=GSE152678) (glyphosate). The genoeme sequences could be found in [NCBI](https://www.ncbi.nlm.nih.gov/genome/annotation_euk/Rattus_norvegicus/106/). 



## Run the code

### First Step: Download the datasets. 

You can download the data using the likns provided in data availibilty section. The toxicant datasets are used for extracting p-value and the associated starting point of each region and the Rattus dataset is used for extracting the DNA sequence. Please put the datasets under the same directory or modify the `DATADIR` variable according to your data path.

### Second Step: Choose a threshold 

The genome was divided into 1000bp regions, and DMRs with a specific pathology were identified. A p-value was calculated for each of the 1000bp regions indicating the probability the region is not a DMR (non-DMR). Those regions whose p-value 10<sup>-15</sup> comprise the final set of DMRs, which constitute the positive examples (DMRs) in the set of training examples used to train the hybrid model, as described in the Methods. A diiferent threshold can be chosen by chagning the value of `threshold` in `hybrid_model.py` and `DL_model.py`.

### Train on your own dataset 

The parameter for the current version of the project is for [1000,5] NumPy one-hot encoding array.  If you want to train a hybrid model on a dataset you can load your Numpy array dataset instead of data and labels and modify the `INPUT_SHAPE` regarding the size of your sequence (e.g. If a sequence is 1000bp and the nucleotides in each region are A, C, T, G, and N the `INPUT_SHPAE` is [1000,5]). 


### Run the code 

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
