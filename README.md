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

The DL-ML-Hybrid system uses Python 3 and requires the following modules: 

* tensorflow
* keras
* biopython
* numpy
* pandas 
* sklearn
* xgboost
* pysster

## Data Availability 

The [Skinner Laboratory](https://skinner.wsu.edu) at Washington State University has produced several datasets based on the rat genome that identify DMRs in the F3 generation male sperm after exposing the F0 generation to one of nine toxicants: atrazine, DDT, glyphosate, vinclozolin, pesticides, dioxin, jet fuel, methoxychlor, and plastics. The data has been deposited into the public database at NCBI under the GEO #s shown below.

The data is assumed to be in CSV format with entries for each 1000bp region of each chromosome. Each entry begins with the chromosome, start and stop values for the region. Next are several attributes about the region, one of which is the "edge.R.p-value" (can be changed in `P_VALUE_HEADER` variable) that contains the probability the region is not a DMR. Note that three of the datasets below (Atrazine, DDT, Vinclozolin) use 100bp regions that would need to be combined into 1000bp regions, and two of the datasets (DDT, Vinclozolin) have four p-values (tnt.edgeR.pv, pnp.edgeR.pv, knk.edgeR.pv, mnm.edgeR.pv), one of which would need to be chosen, before used in the Hybrid code. The other exposures consist of multiple datasets based on observed pathology. Any or all of the datasets can be included.

* Atrazine (100bp): [GSE98683](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98683)
* DDT (100bp, multiple p-values): [GSE114032](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114032)
* Vinclozolin (100bp, multiple p-values): [GSE113785](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113785)
* Jet Fuel: [GSE155922](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE155922)
* Dioxin: [GSE157539](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE157539)
* Pesticides: [GSE158254](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158254)
* Methoxychlor: [GSE158086](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158086)
* Plastics: [GSE163412](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE163412)
* Glyphosate: [GSE152678](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152678)

These datasets are based on v6.0 of the rat genome which can be found [here](https://ftp.ensembl.org/pub/release-104/fasta/rattus_norvegicus/dna/). The genome data is assumed to be in FASTA format and separated into different files by chromosome.

## Running the code

### Step 1: Prepare the datasets

Download the data using the links provided in Data Availabilty section. The toxicant datasets are used for extracting p-values and the associated starting point of each region. The rat genome dataset is used for extracting the DNA sequence. Put the uncompressed datasets under the `data` directory or modify the `DATADIR` variable according to the data path. For each dataset to be processed, include the file name in the `DATASETS` variable. A region is considered DMR if it is DMR in any of the datasets. A region is considered a non-DMR if it is a non-DMR in all datasets. The rat genome FASTA files are assume to have the prefix specified in `RN_PREFIX`, which can be changed if using a different variant.

### Step 2: Choose thresholds

The genome was divided into 1000bp regions, and DMRs with a specific pathology were identified. A p-value was calculated for each of the 1000bp regions indicating the probability the region is not a DMR (non-DMR). Those regions whose p-value is less than 10<sup>-5</sup> comprise the final set of DMRs, the positive examples in the training set. Those regions whose p-value is greater than (1 - 10<sup>-5</sup>) comprise the negative examples (non-DMRs). Different thresholds can be chosen by changing the values of `THRESHOLD_DMR` and `THRESHOLD_NDMR` in `models.py`.

### Step 3: Run the code 

The program has options to specify the type of model, type of training or testing; or you can visualize the model.

* Train the hybrid model (writes model to output directory): 

`python models.py --model hybrid --task train`

* Test the hybrid model (reads last trained model from output directory): 

`python models.py --model hybrid --task test`

* Train and test the hybrid model using k-fold cross-validation (default k=5): 

`python models.py --model hybrid --task kfold --folds=3`

* Train the DL model (writes model to output directory):

`python models.py --model DL --task train`

* Test the DL model (reads last trained model from output directory):

`python models.py --model DL --task test`

* Train and test the DL model using k-fold cross-validation (default k=5):

`python models.py --model DL --task kfold --folds=3`

* Visualize second convolutional layer of the DL model (reads last trained model from output directory):

`python models.py --visualize`

* Help on usage:

`python models.py --help`

### Train on your own dataset 

The parameter for the current version of the project is for [1000,5] NumPy one-hot encoding array. To train a hybrid model on a different dataset, load your Numpy array dataset instead of data and labels and modify the `INPUT_SHAPE` regarding the size of your sequence. E.g., if a sequence is 500bp and the nucleotides in each region are A, C, T, G, and N the `INPUT_SHAPE` is [500,5].

