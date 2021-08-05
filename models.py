# models.py
#
# Contains both the DL model and the DL-ML-Hybrid model.
# See main() function for options.
#
# Written by Pegah Mavaie & Larry Holder, Washington State University

import os
import sys
import argparse
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import KFold

import xgboost as xgb

from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import Flatten, Activation, Dropout, Dense, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.utils import to_categorical

from Bio import SeqIO

from pysster.One_Hot_Encoder import One_Hot_Encoder
import pysster.utils as utils
from pysster.Motif import Motif

# ----- Global variables -----

# Dataset file names to be used for training and testing (below is only example)
DATASETS = [
    #'GSE155922_jetfuel.kidney.results.csv',
    #'GSE157539_kidney.dmr.results.csv',
    #'GSE158086_kidney.dmr.results.csv',
    #'GSE158254_kidney.dmr.results.csv',
    'GSE163412_kidney.dmr.results.csv'
    ]
# Header used in CSV file for column containing p-value, i.e., Prob(non-DMR)
P_VALUE_HEADER = 'edgeR.p.value'
# DATADIR specifies path containing all datasets
DATADIR = './data/'
# Path to store kernel visualization images
OUTPUT_DIR = './output/'
# File name to store trained model
MODEL_LOG_FILE = 'model_log'
# Prefix on the genome sequence FASTA file names (one file per chromosome)
RN_PREFIX = 'Rattus_norvegicus.Rnor_6.0.dna.chromosome.' 
# List of chromosomes to draw from for training and testing sets: ['1','2',...,'20','X','Y'] for rat
CHROMOSOMES = ['1']
# A threshold for choosing a p-value for sampling DMRs: edgeR.p.value < threshold_dmr
THRESHOLD_DMR = 1e-5
# A threshold for choosing a p-value for sampling non-DMRs: edgeR.p.value > (1 - threshold_ndmr)
THRESHOLD_NDMR = 1e-5
# Threshold on fraction of DNA sequence that consists of CGs to be a CG island
THRESHOLD_CG_ISLAND = 0.2
# Input_shape is used for the "input_shape" parameter of the first layer of DL network.
# It is a 2D array where the first element shows the length of the input sequence, and
# the second element is the number of characters used for the representation (one-hot encoding).
INPUT_SHAPE = [1000,5]


def extract_dna_sequences(dna_seq, starts, seq_len):
    """
    Extract sequences from dna_seq of length seq_len according to starts
    Parameters
    ----------
    @param dna_seq: DNA sequence 
    @param starts: Start points for each sequence 
    @param seq_len: Length of each sequecne
    """
    sequences = []
    for start in starts:
        sequence = dna_seq.seq[int(start):int(start + seq_len)]
        if (len(sequence) == seq_len):
            sequences.append(sequence)
    return sequences


def create_train_test_index(data_seq, labels, test_rate):
    """
    Divide data into training and testing sets according to test_rate.
    Parameters
    ----------
    @param data_seq: data samples
    @param labels: labels for the dataset 
    @param test_rate = fraction of data for testing
    """
    # Create a set of random indices
    shuffled_indices = np.random.permutation (len (data_seq))
    test_set_size = int (len (data_seq) * test_rate)
    test_set_indices = shuffled_indices[:test_set_size]
    train_set_indices = shuffled_indices[test_set_size:]
    # Split training data samples
    train_X = data_seq[train_set_indices]
    train_ground = labels[train_set_indices]
    # Split testing samples
    test_X = data_seq[test_set_indices]
    test_ground = labels[test_set_indices]
    return train_X, train_ground, test_X, test_ground


def build_dl_model(input_shape, nb_classes=2):
    """
    Create a deep learning architecture, compile and return the model
    
    Parameters
    ----------
    @param input_shape: The tensor shape fed to first layer
    @param nb_classes: number of classes
    """
    model = Sequential ()
    # Block 1
    model.add (Conv1D (filters=32, kernel_size=20, padding='same', input_shape=input_shape))
    model.add (BatchNormalization ())
    model.add (Activation ('relu'))
    model.add (Conv1D (filters=32, kernel_size=20, padding='valid'))
    model.add (BatchNormalization ())
    model.add (Activation ('relu'))
    model.add (MaxPooling1D (pool_size=2))
    model.add (Dropout (0.45))
    # Block 2
    model.add (Conv1D (filters=64, kernel_size=20, padding='same'))
    model.add (BatchNormalization ())
    model.add (Activation ('relu'))
    model.add (Conv1D (filters=64, kernel_size=20, padding='valid'))
    model.add (BatchNormalization ())
    model.add (Activation ('relu'))
    model.add (MaxPooling1D (pool_size=2))
    model.add (Dropout (0.45))
    # Dense layer
    model.add (Flatten ())
    model.add (Dense (256))
    model.add (BatchNormalization ())
    model.add (Activation ('relu'))
    model.add (Dropout (0.5))
    model.add (Dense (128))
    model.add (BatchNormalization ())
    model.add (Activation ('relu'))
    model.add (Dropout (0.5))
    model.add (Dense (nb_classes))
    model.add (Activation ('sigmoid'))

    opt = Adam (learning_rate=0.01, decay=0.0005)

    model.compile (loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
    return model


def feature_extractor_from_dl_model(model, data_seq, layer_num):
    """
    Create a new representation of the data by extracting the output of the given DL layer_num
    Parameters
    ----------
    @param model: DL model 
    @param data_seq: DNA sequences for extracting DL representation 
    @param layer_num: the output of this layer is used as feature representation 
    """
    # with a Sequential model
    get_layer_output = K.function ([model.layers[0].input], [model.layers[layer_num].output])
    layer_outs = get_layer_output ([data_seq])[0]
    # change the data from list into numpy format
    out_array = np.array (layer_outs)
    # We can use flatten only if we use a non-dense layer output
    out_array = out_array.flatten ()
    # reshape the output into a 2d-array
    out_array = np.reshape (out_array, (data_seq.shape[0], -1))
    return out_array


def fit_ml_classifier(out_array, labels, classifier_case='XGB'):
    """
    Create a non-DL classifier for the classification task, use classifer_case to choose between RF and XGB
    The classifier is fit given the data and labels
    Parameters
    ----------
    @param out_array: Data with DL representation for training ML model 
    @param labels: labels
    @param classifier_cases: 'XGB': XGBoost classifier 'RF': RandomForest classifier 
    """
    if classifier_case == 'XGB':
        print ('ML classifier: XGB')
        learner = xgb.XGBClassifier (use_label_encoder=False)
        learner.fit (out_array, labels)
    if classifier_case == 'RF':
        print ('ML classifier: RF')
        learner = RandomForestClassifier ()
        learner.fit (out_array, labels)
    return learner


def extract_starts_from_dataset(file_path, chrm_index, threshold, dmrs):
    """
    Extract the start of DMRs or non-DMRs from the data in file_path
    for given chromosome based on the threshold.
    Parameters
    ----------
    @param file_path: file path for extracting indices 
    @param chrm_index: chromosome for which to extract indices
    @param threshold: threshold for choosing DMRs or non-DMRs
    @param dmrs: A boolean parameter for select DMRs (True) or non-DMRs (False)
    """
    global P_VALUE_HEADER
    # Read the dataset
    pdData = pd.read_csv (file_path, dtype={'chr':'string'}, index_col=False)
    pdData = pdData[pdData['chr'] == chrm_index]
    # Extract region starts based on whether we want DMR or non-DMRs samples
    if dmrs:
        starts = pdData.loc[pdData[P_VALUE_HEADER] <= threshold]['start']
    else:
        starts = pdData.loc[pdData[P_VALUE_HEADER] >= (1 - threshold)]['start']
    return np.array(starts)

# TODO: K.gradients not supported
def optimize_input(model, layer_name, node_index, input_data, lr, steps):
    model_input = model.layers[0].input
    loss = K.max(model.get_layer(layer_name).output[...,node_index])
    grads = K.gradients(loss, model_input)[0]
    grads = K.l2_normalize(grads, axis = 1)
    iterate = K.function([model_input, K.learning_phase()], [loss, grads])
    for _ in range(steps):
        loss_value, grads_value = iterate([input_data, 0])
        input_data += grads_value * lr
    return input_data[0], loss_value > 2


# TODO
def get_optimized_input(model, data, layer_name, node_index, boundary, lr, steps, colors_sequence, colors_structure):
    global INPUT_SHAPE
    for _attempt in range(5):
        input_data = np.random.uniform(-boundary, +boundary, (1, INPUT_SHAPE[0], INPUT_SHAPE[1]))
        input_data, success = optimize_input(model, layer_name, node_index, input_data, lr, steps)
        if success: break
    if not success:
        print("Warning: loss did not converge for node {} in layer '{}'".format(node_index, layer_name))
    input_data = np.apply_along_axis(utils.softmax, 1, input_data)
    if not data.is_rna:
        return [Motif(data.one_hot_encoder.alphabet, pwm = input_data).plot(colors_sequence, scale=0.25)]

# TODO
def visualize_optimized_inputs(model, data, layer_name, output_file, bound=0.1, lr=0.02, steps=600, colors_sequence={'A': '#00CC00', 'C': '#0000CC', 'G': '#FFB300', 'T': '#CC0000'}, colors_structure={}, nodes=None):
    if nodes == None:
        nodes = list (range (model.get_layer (layer_name).output_shape[-1]))
    motif_plots = []
    for node in nodes:
        #print ("Optimize node {}...".format (node))
        motif_plots += get_optimized_input (model, data, layer_name, node, bound, lr, steps, colors_sequence, colors_structure)
    utils.combine_images (motif_plots, output_file)


def deciding_ndmrs(ndmrs, dna_sequence, case_criteria = 1):
    """
    Returns the set of non-dmr starts in the genome (dna_sequence) according to the case-criteria,
    which determines which non-dmrs to include from given threshold-based ndmrs and regions which
    have no-CGs or are CG islands.
    Parameters
    ----------
    @param ndmrs: threshold-based non-DMRs for a chromosome
    @param dnq_sequence: DNA sequence for the chromosome
    @param case_criteria:
        case_criteria = 1 : threshold-based non-DMRs
        case_criteria = 2 : noCGs and CG-islands
        case_criteria = 3 : noCGs and CG-islands and threshold-based non-DMRs
    """
    global INPUT_SHAPE, THRESHOLD_CG_ISLAND
    region_len = INPUT_SHAPE[0]
    cg_island_threshold = int(THRESHOLD_CG_ISLAND * region_len)
    if case_criteria == 1:
        return ndmrs
    num_regions = len(dna_sequence) // INPUT_SHAPE[0]
    no_cgs = []
    cg_islands = []
    for i in range(num_regions):
        start = i * INPUT_SHAPE[0]
        region_seq = dna_sequence[start:(start + INPUT_SHAPE[0])]
        num_cgs = region_seq.count("CG")
        if num_cgs == 0:
            no_cgs.append(start+1)
        if num_cgs > cg_island_threshold:
            cg_islands.append(start+1)
    cg_ndmrs = np.union1d(np.array(no_cgs),np.array(cg_islands))
    if case_criteria == 2:
        return cg_ndmrs
    return np.union1d(ndmrs,cg_ndmrs)


def create_one_hot_encoder_sequences(dna_seqs):
    """
    One-hot encoding representation of DNA samples
    Parameters
    ----------
    @param dna_seqs: DNA sequences
    """
    one_hot = One_Hot_Encoder ("ACGTN")
    one_hot_data = np.zeros ((len (dna_seqs), INPUT_SHAPE[0],  INPUT_SHAPE[1]))
    for i in range (0, len(dna_seqs)):
        x = one_hot.encode (dna_seqs[i])
        one_hot_data[i] = x
    return one_hot_data


def create_data_and_labels(chrm_index):
    global DATASETS, DATADIR, THRESHOLD_DMR, THRESHOLD_NDMR, LEAST_SET
    """
    Create data and labels for the given chromosome.
    """
    print('Generating data for chromosome ' + chrm_index + '...')
    # Determine DMRs as the union of DMRs in each dataset
    dmrs = extract_starts_from_dataset (DATADIR + DATASETS[0], chrm_index, THRESHOLD_DMR, True)
    for dataset in DATASETS[1:]:
        new_dmrs = extract_starts_from_dataset (DATADIR + dataset, chrm_index, THRESHOLD_DMR, True)
        dmrs = np.union1d(dmrs, new_dmrs)
    # Determine non-DMRs as the intersection of non-DMRs in each dataset
    ndmrs = extract_starts_from_dataset (DATADIR + DATASETS[0], chrm_index, THRESHOLD_NDMR, False)
    for dataset in DATASETS[1:]:
        new_ndmrs = extract_starts_from_dataset (DATADIR + dataset, chrm_index, THRESHOLD_NDMR, False)
        ndmrs = np.intersect1d(ndmrs, new_ndmrs)
    # Parse through the specific chromosome files to obtain the DNA sequences
    records = SeqIO.parse (DATADIR + RN_PREFIX + chrm_index + '.fa', "fasta")
    first_rec = list (records)[0]
    # Determine final set of non-dmrs (starts of regions)
    ndmrs = deciding_ndmrs(ndmrs, first_rec.seq)
    # Extract DNA sequences for DMRs and non-DMRs
    dmr_seqs = extract_dna_sequences (first_rec, dmrs,  INPUT_SHAPE[0])
    ndmr_seqs = extract_dna_sequences (first_rec, ndmrs,  INPUT_SHAPE[0])
    # Convert to one hot encoding
    dmr_data = create_one_hot_encoder_sequences (dmr_seqs)
    ndmr_data = create_one_hot_encoder_sequences (ndmr_seqs)
    # Stack the data on top of one another
    labels = np.concatenate ((np.zeros (len (ndmr_data)), np.ones (len (dmr_data))))
    data = np.concatenate ((np.array (ndmr_data), np.array (dmr_data)))
    return data, labels


def visualization_for_a_node(data, labels, model, layer_name="conv1d_2"):
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    f1 = open(OUTPUT_DIR + "output_positive.fasta", 'w')
    f2 = open(OUTPUT_DIR + "output_negative.fasta", 'w')
    for i in range(0, len(data)):
        dna_seq = ''.join(['ACGTN'[k] for k in np.argmax(data[i], axis=1)])
        if labels[i] == 1:
            f1.write(">1 \n")
            f1.write(dna_seq + "\n")
        else:
            f2.write(">1 \n")
            f2.write(dna_seq + "\n")
    f1.close()
    f2.close()
    from pysster.Data import Data
    from pysster.Model import Model
    pysster_data = Data([OUTPUT_DIR + 'output_negative.fasta', OUTPUT_DIR + 'output_positive.fasta'], "ACGTN")
    visualize_optimized_inputs(model, pysster_data, layer_name, OUTPUT_DIR + 'pes_conv1.png', bound=0.4)

def fit_dl_model(train_X, train_ground, batch_size=64, nb_classes=2, nb_epoch=30):
    global INPUT_SHAPE, OUTPUT_DIR, MODEL_LOG_FILE
    # Create the model and fit it with the data
    model = build_dl_model (INPUT_SHAPE, nb_classes)
    reduce_lr = ReduceLROnPlateau ('val_loss', 0.5, 5, verbose=0)
    stopper = EarlyStopping ('val_loss', patience=5)
    checkpoints = ModelCheckpoint (OUTPUT_DIR + MODEL_LOG_FILE, "val_loss", save_best_only=True)
    callbacks = [reduce_lr, stopper, checkpoints]
    train_ground_cat = to_categorical (train_ground, num_classes=nb_classes)
    model.fit (train_X, train_ground_cat, batch_size, validation_split=0.2, epochs=nb_epoch, shuffle=True, callbacks=callbacks)
    return model


def evaluate_model(model, out_array, test_ground):
    """
    Evaluate the performance of the model

    Parameters
    ----------
    @param model: The trained model
    @param out_array: testing data
    @param test_ground: testing label
    """
    prediction = model.predict (out_array)
    if (not np.isscalar(prediction[0])):
        prediction = np.argmax (prediction, axis=1)
    print(classification_report(test_ground, prediction))


def train_dl_model(data_seq, labels, split=0.2):
    """
    Train and evaluate a dl model
    Parameters
    ----------
    @param labels: labels
    @param data: whole dataset
    """
    train_X, train_ground, test_X, test_ground = create_train_test_index (data_seq, labels, split)
    # Create the model and fit it with the data
    model = fit_dl_model(train_X, train_ground)
    evaluate_model(model, test_X, test_ground)


def train_hybrid_model(data_seq, labels, layer_num=5, split=0.2):
    """
    Train and evaluate a hybrid model
    Parameters
    ----------
    @param data_seq: whole dataset of DNA sequences
    @param labels: labels
    @param layer_num: layer from which to extract features
    """
    train_X, train_ground, test_X, test_ground = create_train_test_index (data_seq, labels, split)
    # Create the model and fit it with the data
    model = fit_dl_model(train_X, train_ground)
    # Redefine the feature representation for feeding to ML classifier
    out_array = feature_extractor_from_dl_model (model, train_X, layer_num)
    # Train an XGB
    xgb_classifier = fit_ml_classifier (out_array, train_ground, classifier_case='XGB')
    out_array = feature_extractor_from_dl_model (model, test_X, layer_num)
    evaluate_model(xgb_classifier, out_array, test_ground)


def dl_kfold_cross_validation(data, labels, folds=5):
    accuracy_list = []
    f1_score_list = []
    precision_list = []
    recall_list = []
    kf = KFold(n_splits=folds, shuffle=True)
    fold_num = 0
    for train_index, test_index in kf.split(data):
        fold_num += 1
        print('Fold ' + str(fold_num) + ' of ' + str(folds))
        train_X = data[train_index]
        test_X = data[test_index]
        train_ground = labels[train_index]
        test_ground = labels[test_index]
        model = fit_dl_model(train_X, train_ground)
        # Evaluate the performance of the model
        prediction = model.predict (test_X)
        prediction = np.argmax (prediction, axis=1)
        accuracy_list.append (accuracy_score (test_ground, prediction))
        f1_score_list.append (f1_score (test_ground, prediction))
        precision_list.append (precision_score (test_ground, prediction))
        recall_list.append (recall_score (test_ground, prediction))
    print ('accuracy for DL: ', sum (accuracy_list) / len (accuracy_list))
    print ('F1 score for DL: ', sum (f1_score_list) / len (accuracy_list))
    print ('precision DL: ', sum (precision_list) / len (accuracy_list))
    print ('recall for DL: ', sum (recall_list) / len (accuracy_list))


def hybrid_kfold_cross_validation(data, labels, folds=5, layer_num=5):
    accuracy_list = []
    f1_score_list = []
    precision_list = []
    recall_list = []
    kf = KFold(n_splits=folds, shuffle=True)
    fold_num = 0
    for train_index, test_index in kf.split(data):
        fold_num +=1
        print('Fold ' + str(fold_num) + ' of ' + str(folds))
        train_X = data[train_index]
        test_X = data[test_index]
        train_ground = labels[train_index]
        test_ground = labels[test_index]
        model = fit_dl_model(train_X, train_ground)
	 	# Redefine the feature representation for feeding to ML classifier
        out_array = feature_extractor_from_dl_model (model, train_X, layer_num)
        # Train an XGB
        xgb_classifier = fit_ml_classifier (out_array, train_ground, classifier_case='XGB')
        out_array = feature_extractor_from_dl_model (model, test_X, layer_num)
        # Evaluate the performance of the model
        prediction = xgb_classifier.predict (out_array)
        accuracy_list.append (accuracy_score (test_ground, prediction))
        f1_score_list.append (f1_score (test_ground, prediction))
        precision_list.append (precision_score (test_ground, prediction))
        recall_list.append (recall_score (test_ground, prediction))
    print ('accuracy for Hybrid: ', sum (accuracy_list) / len (accuracy_list))
    print ('F1 score for Hybrid: ', sum (f1_score_list) / len (accuracy_list))
    print ('precision Hybrid: ', sum (precision_list) / len (accuracy_list))
    print ('recall for Hybrid: ', sum (recall_list) / len (accuracy_list))


def create_dataset():
    data, labels = create_data_and_labels(CHROMOSOMES[0])
    for chr in CHROMOSOMES[1:]:
        data1, labels1 = create_data_and_labels(chr)
        data = np.concatenate((data, data1))
        labels = np.concatenate((labels, labels1))
    data_seq = data.astype (float)
    labels = labels.astype (int)
    # Check that #pos_egs and #neg_egs are both non-zero
    a, c = np.unique (labels, return_counts=True)
    print ('label distribution: ', a, c)
    if len(a) == 1:
        print('  only one label represented...exiting.')
        sys.exit()
    return data_seq, labels


def main():
    global CHROMOSOMES, OUTPUT_DIR, MODEL_LOG_FILE
    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, choices=['DL','hybrid'])
    parser.add_argument('--task', dest='task', type=str, choices=['train','test','kfold'])
    parser.add_argument('--folds', dest='folds', type=int, default=5)
    parser.add_argument('--visualize', dest='visualize', default=False, action='store_true')
    args = parser.parse_args()
    if args.model == 'hybrid' and args.task == 'train':
        data, labels = create_dataset()
        train_hybrid_model(data, labels)
    elif args.model == 'hybrid' and args.task == 'test':
        model = load_model(OUTPUT_DIR + MODEL_LOG_FILE)
        data, labels = create_dataset()
        evaluate_model(model, data, labels)
    elif args.model == 'hybrid' and args.task == 'kfold':
        data, labels = create_dataset()
        hybrid_kfold_cross_validation(data, labels, folds=args.folds)
    elif args.model == 'DL' and args.task == 'train':
        data, labels = create_dataset()
        train_dl_model(data, labels)
    elif args.model == 'DL' and args.task == 'test':
        model = load_model(OUTPUT_DIR + MODEL_LOG_FILE)
        data, labels = create_dataset()
        evaluate_model(model, data, labels)
    elif args.model == 'DL' and args.task == 'kfold':
        data, labels = create_dataset()
        dl_kfold_cross_validation(data, labels, folds=args.folds)
    elif args.visualize:
        model = load_model(OUTPUT_DIR + MODEL_LOG_FILE)
        data, labels = create_dataset()
        visualization_for_a_node(data, labels, model)
    else:
        parser.print_help()
        

if __name__ == "__main__":
    main()
