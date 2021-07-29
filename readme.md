# DL-ML-Hybrid Approach

In this project, we introduce a hybrid DL-ML approach that uses a deep neural network for extracting features and a non-DL classifier for classification tasks. The approach is targeted toward DNA sequence-based classification tasks. This particular implementation includes code and data designed to predict environmentally responsive transgenerational differential DNA methylated regions (DMRs). 

# Overview

The goal is to build a classification model that takes a region of the genome as input and predicts the region’s susceptibility to develop an environmentally induced transgenerational alteration in differential DNA methylation regions (DMRs) in the F3 generation from an ancestrally exposed F0 generation (great grandmother). 

The DL network is used to learn features, which are extracted and used as the input features for an XGBoost model to build a classifier for predicting DMRs (or any other classification task). The approach defaults to using a 1000bp DNA sequence as input, but can be used for any DNA classification problem with a minor modification in the shape of the input data. The main features of this project are: 

* Multi-class classification using deep learning and machine learning algorithm.
* Interpretation of learned structures (motifs) in DNA sequences using a modified version of the [pysster](https://github.com/budach/pysster) package.
* Train a hybrid model on DNA sequences using a one-hot encoding representation. 
* Test a pre-trained model on a dataset (Input: one-hot encoding NumPy array of DNA sequences, labels)
* Train and test 5-fold cross-validation and report the classification results on the dataset. 
* Visualize the kernels of convolutional and dense layers.
* Hand-crafted features can also be added to XGBoost by modifying the input argument of the `fit_ml_classifier` function.

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

The Skinner laboratory at Washington State University has produced several datasets based on the rat genome that identify DMRs in the F3 generation male sperm after exposing the F0 generation to one of nine toxicants: atrazine, DDT, glyphosate, vinclozolin, pesticides, dioxin, jet fuel, methoxychlor, and plastics.

All molecular data has been deposited into the public database at NCBI under the GEO #s: [GSE113785](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113785) (vinclozolin), [GSE114032](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114032) (DDT), [GSE98683](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98683) (atrazine), [GSE155922](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE155922) (jet fuel), [GSE157539](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE157539) (dioxin), [GSE158254](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158254) (pesticides), [GSE158086](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158086) (methoxychlor), [GSE163412](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE163412) (plastics), and [GSE152678](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152678) (glyphosate). These datasets are based on v6.0 of the rat genome which can be found [here](https://ftp.ensembl.org/pub/release-104/fasta/rattus_norvegicus/dna/). 

## Run the code

### Step 1: Download the datasets. 

Download the data using the links provided in Data Availabilty section. The toxicant datasets are used for extracting p-values and the associated starting point of each region. The rat genome dataset is used for extracting the DNA sequence. Put the uncompressed datasets under the `data` directory or modify the `DATADIR` variable in `models.py` according to the data path.

### Step 2: Choose a threshold 

The genome was divided into 1000bp regions, and DMRs with a specific pathology were identified. A p-value was calculated for each of the 1000bp regions indicating the probability the region is not a DMR (non-DMR). Those regions whose p-value is less than 10<sup>-15</sup> comprise the final set of DMRs, the positive examples in the training set. Those regions whose p-value is greater than (1 - 10<sup>-8</sup>) comprise the negative examples (non-DMRs). Different thresholds can be chosen by changing the value of `threshold` and `threshold_ndmr` in `models.py`.

### Step 3: Run the code 

* Train a hybrid model: 

`python hybrid_model.py train-hybrid`

* Test a hybrid model: 

`python hybrid_model.py test-hybrid`

* Generate a classification report of the hybrid method on a dataset using 5-fold cross-validation: 

`python hybrid_model.py 5-fold-hybrid`

* Visualize a layer of the DL network:

`python hybrid_model.py visualize`

* Train a DL model:

`python DL_model.py train-DL`

* Test a DL model:

`python DL_model.py test-DL`

* Generate a classification report of a DL network on a dataset using 5-fold cross-validation:

`python DL_model.py 5-fold-DL`

### Train on your own dataset 

The parameter for the current version of the project is for [1000,5] NumPy one-hot encoding array. To train a hybrid model on a different dataset, load your Numpy array dataset instead of data and labels and modify the `INPUT_SHAPE` regarding the size of your sequence. E.g., if a sequence is 500bp and the nucleotides in each region are A, C, T, G, and N the `INPUT_SHPAE` is [500,5]).

