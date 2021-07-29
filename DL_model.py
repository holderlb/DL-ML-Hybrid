# -*- coding: utf-8 -*-

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import pandas as pd
import numpy as np
import keras
from pysster.One_Hot_Encoder import One_Hot_Encoder
import pysster.utils as utils
from pysster.Motif import Motif
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Reshape, Flatten, TimeDistributed, GRU, \
    Bidirectional, Input, Dropout, Dense, concatenate, Activation
from keras.layers import Conv1D, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Convolution2D, GlobalAveragePooling1D, \
    UpSampling1D, MaxPooling1D, Reshape, Flatten, BatchNormalization, LSTM, TimeDistributed, GRU, Bidirectional, Input, \
    Multiply
from keras import backend as K
from keras.models import Model
from Bio import SeqIO
from sklearn.utils import class_weight
from pysster.One_Hot_Encoder import One_Hot_Encoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
from keras.initializers import RandomUniform
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras import backend as K
from keras.constraints import max_norm
import sys
import xgboost as xgb
from keras.models import load_model


#DATADIR parameters are specified for the proposed datasets the original file paths 
DATADIR = './'
#The path for kernel visualization 
output_folder = "./example_artifical/"
least_set = 0
# To train or test a data on a specific chromosome
chrm_index = 'X'
# A threshold for choosing a p-value for sampling DMRs 
threshold = 0.000000000000001
# A threshold for choosing a p-value for sampling DMRs 
threshold_ndmr = 0.00000001

#Input_shape is used for the "input_shape" parameter of the first layer of DL network and it is a 2D array that first element shows the lenght of the sequence and second element is the number of character that is used for the representation (one-hot encoding representation)
INPUT_SHAPE = [1000,5]

def extract_dna_sequences(record, label_indices, seq_len):
    """
    Extract sequence from the genome using dataset
    Parameters
    ----------
    @param record: FASTA file 
    @param label_indices: The start points for each sequence 
    @seq_len: The lengh of each sequecne
    """
    x_data = []
    for i in label_indices:
        if not np.isnan(i):
            if (len (record.seq[int(i):int(i + seq_len)]) == seq_len):
                x_data.append ((record.seq[int(i):int(i + seq_len)]))
    return x_data



def create_train_test_index(data_seq, labels, test_rate):
    """
    Create a train_and_test_Split function that can be used with any dimention
    Parameters
    ----------
    @param data_seq: data samples
    @param labels: labels for thte dataset 
    @param test_rate = The ratio of our testing data
    """
    # Create a set of random number
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


def build_dl_model(input_shape=[1000, 5], batch_size=64, nb_classes=2, nb_epoch=30):
    """
    Create a deep learning architecture, complie the model and return the model
    
    Parameters
    ----------
    @param input_shape: The tensor shape fed to first layer
    @param batch_size: The batch_size in each epoch
    @param nb_classes: number of classes
    @param nb_epoch: number of epochs
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

    opt = keras.optimizers.Adam (lr=0.01, decay=0.0005)

    model.compile (loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
    return model


def feature_extractor_from_dl_model(model, data_seq, layer_num):
    """
    Create a new representation of the data, extract the output of the given DL layer_num
    Parameters
    ----------
    @param model: DL model 
    @param data_seq: DNA sequences for extracting DL representation 
    @param layer_num: the output of this layer is used as feature representation 
    """
    inp = model.input  # input placeholder

    # with a Sequential model
    get_layer_output = K.function ([model.layers[0].input],
                                        [model.layers[layer_num].output])
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
    Create a non-DL classifier for the classification task, use classifer_case to choose between RF, and XGB
    The classifier is fitted given the data and labels
    Parameters
    ----------
    @param out_array: Data with DL representation for training ML model 
    @param labels: labels
    @param classifier_cases: 'XGB': XGBoost classifier 'RF': RandomForest classifier 
    """
    if classifier_case == 'XGB':
        print ('The ML classifier is XGB')
        learner = xgb.XGBClassifier ()
        learner.fit (out_array, labels)
    if classifier_case == 'RF':
        learner = RandomForestClassifier ()
        learner.fit (out_array, labels)
    return learner


def evaluate_performace_model(model, out_array, labels):
    """
    Compute the accuracy and print-out the classification report and the overlap between training samples
    Parameters
    ----------
    @param model: model for prediction 
    @param out_array: testing data
    @param labels: labels
    """
    prediction = model.predict (out_array)
    print (sklearn.metrics.classification_report (labels, prediction))
    print ('accuracy for XGB limited depth: ', accuracy_score (labels, prediction))
    overlap = labels + prediction
    print ('The number of pridected and overlap: ', (np.count_nonzero (overlap == 2)))


def extract_indices_from_dataset(file_path, chrm_index, threshold, dmrs):
    """
    Extract the indices of DMRs or non-DMRs based on the threshold
    Parameters
    ----------
    @param file_path: file path for extracting labels 
    @param chrm_index: Chromosome 
    @param threshold: threshold for choosing DMRs 
    @param dmrs: A boolian parameter for select DMRs (True) or non-DMRs (False)
    """
    # Read the dataset
    pdData = pd.read_csv (file_path, index_col=False)
    first_chr = pdData[pdData['chr'] == chrm_index]
    # check whether we want to check the DMR or non-DMRs samples (dmrs), then extract the associated indices
    if dmrs:
        indices = first_chr.loc[first_chr['edgeR.p.value'] <= threshold].index.values
    else:
        indices = first_chr.loc[first_chr['edgeR.p.value'] >= (1 - threshold)].index.values
    return indices


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


def softmax(x):
    x = np.exp(x - np.max(x))
    return x / x.sum(axis = 0)



def extract_pwm( input_data, annotation, alphabet):
    pwm = []
    for char in alphabet:
        idx = [m.start() for m in re.finditer(re.escape(char), annotation)]
        pwm.append(np.sum(input_data[:,idx], axis = 1))
    return np.transpose(np.array(pwm))


def get_optimized_input( model, data, layer_name, node_index, boundary, lr, steps, colors_sequence, colors_structure):
    for _attempt in range(5):
        input_data = np.random.uniform(-boundary, +boundary,
                                       (1,1000, 5))
        input_data, success = optimize_input(model, layer_name, node_index, input_data, lr, steps)
        if success: break
    if not success:
        print("Warning: loss did not converge for node {} in layer '{}'".format(node_index, layer_name))
    input_data = np.apply_along_axis(utils.softmax, 1, input_data)
    if not data.is_rna:
        return [Motif(data.one_hot_encoder.alphabet, pwm = input_data).plot(colors_sequence, scale=0.25)]



def visualize_optimized_inputs(model, data, layer_name, output_file, bound=0.1, lr=0.02, steps=600, colors_sequence={'A': '#00CC00', 'C': '#0000CC', 'G': '#FFB300', 'T': '#CC0000'}, colors_structure={}, nodes=None):

    if nodes == None:
        nodes = list (range (model.get_layer (layer_name).output_shape[-1]))
    motif_plots = []
    for node in nodes:
        print ("Optimize node {}...".format (node))
        motif_plots += get_optimized_input (model, data, layer_name, node, bound,
                                                lr, steps, colors_sequence, colors_structure)
    utils.combine_images (motif_plots, output_file)



def chr_datasets(file_path, chrm_index):
    """
    Extract a dataframe for an specific threshold 
    Parameters
    ----------
    @param file_path: file path 
    @param chrm_index: Chromosome number 
    """
    pdData = pd.read_csv (file_path, index_col=False)
    first_chr = pdData[pdData['chr'] == chrm_index]
    return first_chr


def deciding_ndmrs(x_ndmr, ndmrs, result, least_set, case_criteria):
    """
    given the indices and the case criteria, this function decide how to choose non-DMRs 
    Parameters
    ----------
    @param x_ndmrs: DNA sequences for non-DMRs 
    @param ndmrs: threshold-based non-DMRs
    @param results: the number of different exposures woth p-value greater than the threshold 
    @param least_set: The number of exposure that a sample is non-DMR in at least this value exposure
    @param case_criteria: 
        case_criteria = 1 : noCGs or CG-islands or threshold-based non-DMRs 
        case_criteria = 2 : noCGs or CG-islands 
        case_criteria =3 : noCGs or CG-islands 
    """
    cg_counter_ndmrs = np.ones ((len (x_ndmr)))
    if case_criteria == 1:
        for i in range (0, len (x_ndmr)):
            tmp_count = ''.join (x_ndmr[i])
            if tmp_count.count ('CG') == 0 or tmp_count.count ('CG') >= 150 or i in ndmrs:
                cg_counter_ndmrs[i] = 0
        x_ndmr = np.array (x_ndmr)
        return x_ndmr[np.where (cg_counter_ndmrs == 0)]
    elif case_criteria == 2:
        for i in range (0, len (x_ndmr)):
            tmp_count = ''.join (x_ndmr[i])
            if tmp_count.count ('CG') == 0 or tmp_count.count ('CG') >= 150:
                cg_counter_ndmrs[i] = 0
        x_ndmr = np.array (x_ndmr)
        return x_ndmr[np.where (cg_counter_ndmrs == 0)]
    if case_criteria == 3:
        for i in range (0, len (x_ndmr)):
            tmp_count = ''.join (x_ndmr[i])
            if tmp_count.count ('CG') == 0 or tmp_count.count ('CG') >= 150 or result[i] >= least_set:
                cg_counter_ndmrs[i] = 0
        x_ndmr = np.array (x_ndmr)
        return x_ndmr[np.where (cg_counter_ndmrs == 0)]


def create_one_hot_encoder_sequences(x_ndmr):
    """
    One-hot encoding representation of DNA samples
    Parameters
    ----------
    @param x_ndmrs: DNA sequences
    """

    one_hot = One_Hot_Encoder ("ACGTN")
    ndmr_data = np.zeros ((len (x_ndmr), INPUT_SHAPE.shape[0],  INPUT_SHAPE.shape[1]))
    for i in range (0, len (x_ndmr)):
        x = one_hot.encode (x_ndmr[i])
        ndmr_data[i] = x
    return ndmr_data


def create_binary_ndmrs_union(first_chr, threshold_ndmr):
    """
    create a binary labels 
    Parameters
    ----------
    @param first_chr: dataframe 
    @param threshold_ndmr: threshold for choosing non-DMRs and DMRs 
    """
    ddt = np.array (first_chr['edgeR.p.value'])
    # change it to binary labels
    ddt[np.where (ddt >= (1 - threshold_ndmr))] = 1
    ddt[np.where (ddt != 1)] = 0
    return ddt


def create_ndmrs_from_union_to_intersection(chrm_index, threshold_ndmr):
    """
    This fuction is used when we want to select non-DMRs such that they are non-DMRs in at least sum exposures
    
    Parameters
    ----------
    @param chrm_index: chromosome number 
    @param threshold_ndmr: threshold for choosing non-DMRs and DMRs 
    
    """
    # Read the chrm_datasets
    first_chr = chr_datasets (DATADIR + 'ddt.result.table.W1000.csv', chrm_index)
    first_chr_vin = chr_datasets (DATADIR + 'vinclozolin.result.table.W1000.csv', chrm_index)
    first_chr_atr = chr_datasets (DATADIR + 'atrazine.result.table.W1000.csv', chrm_index)
    first_chr_gly = chr_datasets (DATADIR + 'glyphosate.result.table.W1000.csv', chrm_index)
    first_chr_plas = chr_datasets (DATADIR + 'plasticsVScombinedControl.result.csv', chrm_index)
    first_chr_pes = chr_datasets (DATADIR + 'pesticidesVScombinedControl.result.csv', chrm_index)
    first_chr_dix = chr_datasets (DATADIR + 'pesticidesVScombinedControl.result.csv', chrm_index)
    first_chr_meth = chr_datasets (DATADIR + 'methoxychlorVScombinedControl.result.csv', chrm_index)
    first_chr_jet = chr_datasets (DATADIR + 'jetFuelVScombinedControl.result.csv', chrm_index)
    # create labels for all the exposures
    ddt = create_binary_ndmrs_union (first_chr, threshold_ndmr)
    vin = create_binary_ndmrs_union (first_chr_vin, threshold_ndmr)
    atr = create_binary_ndmrs_union (first_chr_atr, threshold_ndmr)
    gly = create_binary_ndmrs_union (first_chr_gly, threshold_ndmr)
    plas = create_binary_ndmrs_union (first_chr_plas, threshold_ndmr)
    pes = create_binary_ndmrs_union (first_chr_pes, threshold_ndmr)
    dix = create_binary_ndmrs_union (first_chr_dix, threshold_ndmr)
    meth = create_binary_ndmrs_union (first_chr_meth, threshold_ndmr)
    jet = create_binary_ndmrs_union (first_chr_jet, threshold_ndmr)

    results = ddt + vin + atr + gly + plas + pes + dix + meth + jet
    return results


def create_data_and_labels(chrm_index, threshold, threshold_ndmr, least_set):
    """
    Create data and labels for the model
    Parameters
    ----------
    @param chrm_index: chromosome number 
    @param threshold: threshold for choosing DMRs 
    @param least_set: The number of exposure that a sample is non-DMR in at least this value exposure
    @param threshold_ndmr: threshold for choosing non-DMRs and DMRs 
    """
    print (threshold, threshold_ndmr)
    dmrs_ddt = extract_indices_from_dataset (DATADIR + 'ddt.result.table.W1000.csv', chrm_index,
                                             threshold, True)
    dmrs_vin = extract_indices_from_dataset (DATADIR + 'vinclozolin.result.table.W1000.csv',
                                             chrm_index, threshold, True)
    dmrs_atr = extract_indices_from_dataset (DATADIR + 'atrazine.result.table.W1000.csv', chrm_index,
                                             threshold, True)
    dmrs_gly = extract_indices_from_dataset (DATADIR + 'glyphosate.result.table.W1000.csv',
                                             chrm_index, threshold, True)
    dmrs_plas = extract_indices_from_dataset (DATADIR + 'plasticsVScombinedControl.result.csv',
                                              chrm_index, threshold, True)
    dmrs_pes = extract_indices_from_dataset (DATADIR + 'pesticidesVScombinedControl.result.csv',
                                             chrm_index, threshold, True)
    dmrs_jet = extract_indices_from_dataset (DATADIR + 'dioxinVScombinedControl.result.csv',
                                             chrm_index, threshold, True)
    dmrs_dix = extract_indices_from_dataset (DATADIR + 'methoxychlorVScombinedControl.result.csv',
                                             chrm_index, threshold, True)
    dmrs_meth = extract_indices_from_dataset (DATADIR + 'jetFuelVScombinedControl.result.csv',
                                              chrm_index, threshold, True)

    ndmrs_ddt = extract_indices_from_dataset (DATADIR + 'ddt.result.table.W1000.csv', chrm_index,
                                              threshold_ndmr, False)
    ndmrs_vin = extract_indices_from_dataset (DATADIR + 'vinclozolin.result.table.W1000.csv',
                                              chrm_index, threshold_ndmr, False)
    ndmrs_atr = extract_indices_from_dataset (DATADIR + 'atrazine.result.table.W1000.csv',
                                              chrm_index, threshold_ndmr, False)
    ndmrs_gly = extract_indices_from_dataset (DATADIR + 'glyphosate.result.table.W1000.csv',
                                              chrm_index, threshold_ndmr, False)
    ndmrs_plas = extract_indices_from_dataset (DATADIR + 'plasticsVScombinedControl.result.csv',
                                               chrm_index, threshold_ndmr, False)
    ndmrs_pes = extract_indices_from_dataset (DATADIR + 'pesticidesVScombinedControl.result.csv',
                                              chrm_index, threshold_ndmr, False)
    ndmrs_jet = extract_indices_from_dataset (DATADIR + 'dioxinVScombinedControl.result.csv',
                                              chrm_index, threshold_ndmr, False)
    ndmrs_dix = extract_indices_from_dataset (DATADIR + 'methoxychlorVScombinedControl.result.csv',
                                              chrm_index, threshold_ndmr, False)
    ndmrs_meth = extract_indices_from_dataset (DATADIR + 'jetFuelVScombinedControl.result.csv',
                                               chrm_index, threshold_ndmr, False)

    dmrs_overlap1 = np.union1d (dmrs_atr, (np.union1d (dmrs_ddt, dmrs_vin)))
    dmrs_overlap1 = np.union1d (dmrs_overlap1, dmrs_plas)
    dmrs_overlap1 = np.union1d (dmrs_overlap1, dmrs_pes)
    dmrs_overlap1 = np.union1d (dmrs_overlap1, dmrs_meth)
    dmrs_overlap1 = np.union1d (dmrs_overlap1, dmrs_jet)
    dmrs_overlap1 = np.union1d (dmrs_overlap1, dmrs_dix)
    dmrs_overlap = np.union1d (dmrs_overlap1, dmrs_gly)

    ndmrs_overlap1 = np.intersect1d (ndmrs_atr, (np.intersect1d (ndmrs_ddt, ndmrs_vin)))
    ndmrs_overlap1 = np.intersect1d (ndmrs_overlap1, ndmrs_plas)
    ndmrs_overlap1 = np.intersect1d (ndmrs_overlap1, ndmrs_pes)
    ndmrs_overlap1 = np.intersect1d (ndmrs_overlap1, ndmrs_meth)
    ndmrs_overlap1 = np.intersect1d (ndmrs_overlap1, ndmrs_jet)
    ndmrs_overlap1 = np.intersect1d (ndmrs_overlap1, ndmrs_dix)
    ndmrs_overlap = np.intersect1d (ndmrs_overlap1, ndmrs_gly)

   
    # Determine number of DMR and NDMR based on current threshold
    dmrs = dmrs_overlap
    ndmrs = ndmrs_overlap

    first_chr = chr_datasets (DATADIR + 'ddt.result.table.W1000.csv', chrm_index)

    # Parse through the specific chromosome files to obtain the DNA sequences in the form of 1000 bp
    records = SeqIO.parse (
        DATADIR + 'dna_seq/Rattus_norvegicus.Rnor_6.0.dna.chromosome.' + str (chrm_index) + '.fa',
        "fasta")
    first_rec = list (records)[0]
    x = np.array (first_chr.loc[dmrs, 'start'])
    y = np.array (first_chr['start'])

    x_ndmr = extract_dna_sequences (first_rec, y,  INPUT_SHAPE.shape[0])
    x_dmr = extract_dna_sequences (first_rec, x,  INPUT_SHAPE.shape[0])
    ##result_ndmr_binary is used as an input for "deciding_ndmrs" with case_criteria=3
    #result_ndmr_binary = create_ndmrs_from_union_to_intersection (chrm_index, threshold_ndmr)
    result_ndmr_binary = []
    ndmrs_index = np.array (first_chr.loc[ndmrs, 'start'])
    ndmrs_index = (ndmrs_index // 1000) - 1
    x_ndmr = deciding_ndmrs (x_ndmr, ndmrs_index, result_ndmr_binary, least_set, case_criteria=2)

    # One hot encoding DNA sequences
    one_hot = One_Hot_Encoder ("ACGTN")
    ndmr_data = create_one_hot_encoder_sequences (x_ndmr)
    dmr_data = create_one_hot_encoder_sequences (x_dmr)

    # Stack the data on top of one another

    labels = np.concatenate ((np.zeros (len (x_ndmr)), np.ones (len (x_dmr))), axis=0)
    data = np.concatenate ((np.array (ndmr_data), np.array (dmr_data)), axis=0)

    return data, labels


def show_distribution_labels(labels):
    """
    Show label distributuion
    Parameters
    ----------
    @param labels: labels 
    """
    a, c = np.unique (labels, return_counts=True)
    print ('labels distribution: ', a, c)


def inference_hybrid(model, rf, test_x, layer_num):
    """
    Test a batch of data extract features and predict with ML classifier given the trained DL model and trained ML moedl
    
    Parameters
    ----------
    @param model: DL model
    @rf: ML model 
    @test_x: testing data
    @layer_num: The layer used for feature representaiton 
    """
    get_layer_output = K.function ([model.layers[0].input],
                                        [model.layers[layer_num].output])
    layer_outs = get_layer_output ([test_x])[0]
    out_array = layer_outs.flatten ()
    out_array = np.reshape (out_array, (test_x.shape[0], -1))

    predicted_labels = rf.predict (out_array)
    # predicted_labels_index = np.argmax(predicted_labels, axis=1)
    return predicted_labels



def visualization_for_a_node(data, labels, model, layer_name="conv1d_2"):
    labels = labels.astype (int)
    print (data.shape)
    if not os.path.isdir(output_folder):
    os.makedirs(output_folder)


    print("Training data: ", data.shape)

    f = open("./output_positive.fasta", 'w')
    f2 = open("./output_negative.fasta", 'w')
    for i in range(0, len(data)):
        if labels[i] == 1:
            f.write(">1 \n")
            x = ''.join(data[i, :])
            f.write(x + "\n")
        else:
            f2.write(">0 \n")
            x = ''.join(data[i, :])
            f2.write(x + "\n")

    from pysster.Data import Data
    from pysster.Model import Model

    data = Data(['./output_negative.fasta',
             './output_positive.fasta'], "ACGTN")

    visualize_optimized_inputs(model, data, layer_name, output_folder + str(chrm_index)+'pes_conv1.png', bound=0.4)


def fit_dl_model(train_X, train_ground, input_shape=[1000, 5], batch_size=64, nb_classes=2, nb_epoch=30):

    # Create the model and fit it with the data
    model = build_dl_model (input_shape, batch_size, nb_classes, nb_epoch)
    reduce_lr = ReduceLROnPlateau ('val_loss', 0.5, 5, verbose=0)
    stopper = EarlyStopping ('val_loss', patience=5)
    checkpoints = ModelCheckpoint ('./model_log', "val_loss", save_best_only=True)
    callbacks = [reduce_lr, stopper, checkpoints]
    trained_model = model.fit (train_X, to_categorical (train_ground), batch_size, validation_split=0.2, epochs, shuffle=True, callbacks=callbacks) 
    return trained_model



def evaluate_models(classifier, out_array, test_ground):
    """
    Evaluate the performance of the model 
    
    Parameters
    ----------
    @param classiffier: The trained model  
    @param out_array: testing data
    @param test_ground: testing label 
    """
    prediction = classifier.predict (out_array)
    if len(prediction) > 1: 
        prediction = np.argmax (prediction, axis=1)
    print(sklearn.metrics.classification_report(test_ground, prediction))
    accuracy_list.append (accuracy_score (test_ground, prediction))
    f1_score_list.append (f1_score (test_ground, prediction))
    precision_list.append (precision_score (test_ground, prediction))
    recall_list.append (recall_score (test_ground, prediction))





def train_dl_model(data_seq, labels):
    """
    Train a dl model ecaluate the performance and return the hybrid model 
    Parameters
    ----------
    @param labels: labels 
    @param data: whole dataset 
    """
    show_distribution_labels (labels)
    train_X, train_ground, test_X, test_ground = create_train_test_index (data_seq, labels, 0.2)
    # Create the model and fit it with the data

    model = fit_dl_model(train_X, train_ground)
    # Rdedifine the feature representation for feeding to ML classifier
    evaluate_models(model, out_array, test_ground)


def dl_nfold_cross_validation(data_seq, labels, nfold=5):
    show_distribution_labels (labels)
    
    # Create the model and fit it with the data

    model = build_dl_model (input_shape=[1000, 5], batch_size=64, nb_classes=2, nb_epoch=30)
    print (model.summary ())
    accuracy_list = []
    f1_score_list = []
    precision_list = []
    recall_list = []

    for ifold in range (0, (nfold)):
        train_X, train_ground, test_X, test_ground = create_train_test_index (data_seq, labels, 0.2)
        model = fit_dl_model(train_X, train_ground)
        # Rdedifine the feature representation for feeding to ML classifier
     
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







def main():
    
    """
    For testing on a different dataset comment the next line and load the dataset into a one-hot encoding numpy array 
    """
    data, labels = create_data_and_labels (chrm_index, threshold, threshold_ndmr, least_set)
    data_seq = data.astype (float)
    labels = labels.astype (int)
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', dest='option', type=str, default='train',
                        help='actions: train-hybrid, train-DL, test-DL, test-hybrid, 5-fold hybrid, 5-fold DL, or visualize')
    args = parser.parse_args()
    if args.option not in ['train-hybrid', 'train-DL','test-DL', 'test_hybrid' '5-fold-hybrid', '5-fold-DL', 'visualize']:
        print('invalid option: ', args.option)
        print("Please input a option: train DL, test-DL, 5-fold DL, or visualize")
    elif args.option== 'train-DL':
        train_dl_model(data_seq, labels)
    elif args.option== 'test-DL':
        classifier = load_model('model_log')
        evaluate_models(classifier, data_seq, labels)
    elif args.option== '5-fold-DL':
        dl_nfold_cross_validation(data_seq, labels)
    elif args.option== 'visualize':
        visualization_for_a_node(data, labels, model)


if __name__ == "__main__":
    main ()
