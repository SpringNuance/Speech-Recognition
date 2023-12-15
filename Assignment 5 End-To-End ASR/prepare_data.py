import numpy as np
import string

import torch
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence

def load_features(features_path):
    feature_array = []
    features = np.load(features_path, allow_pickle=True)
    
    for feature in features:
        feature_array.append(autograd.Variable(torch.FloatTensor(feature)))

    return feature_array


def load_transcripts(features_path):
    label_array = []
    with open(features_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    for sent in data:
        sent = sent.rstrip()
        label_array.append([sent])

    return label_array


def encode_data():
    chars = list(string.ascii_lowercase)
    additional_chars = ['\'', ' ']
    chars += additional_chars

    char2idx = {'<sos>': 1, '<eos>': 2}
    idx2char = {1: '<sos>', 2: '<eos>'}

    for c in chars:
        if c not in char2idx.keys():
            char2idx[c] = len(char2idx) + 1
            idx2char[len(idx2char) + 1] = c
    
    return char2idx, idx2char


def label_to_idx(labels, char2idx):
    res = []
    for sent in labels:
        temp_sent = []
        sent = sent[0].split()
        for word in sent:
            for char in word:
                temp_sent.append([char2idx[char]])
            temp_sent.append([char2idx[' ']])
        temp_sent.append([char2idx['<eos>']])

        res.append(torch.LongTensor(temp_sent))
    return res


def combine_data(features, indexed_labels):
    res = []
    for i in range(len(features)):
        res.append((features[i], indexed_labels[i]))
    return res


def remove_extra(data, batch_size):
    extra = len(data) % batch_size
    if extra != 0:
        data = data[:-extra][:]
    return data
    

def collate(list_of_samples):
    list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)
    
    input_seqs, output_seqs = zip(*list_of_samples)

    input_seq_lengths = [len(seq) for seq in input_seqs]
    output_seq_lengths = [len(seq) for seq in output_seqs]

    padding_value = 0

    # pad input sequences
    pad_input_seqs = pad_sequence(input_seqs, padding_value=padding_value)

    # pad output sequences
    pad_output_seqs = pad_sequence(output_seqs, padding_value=padding_value) 


    return pad_input_seqs, input_seq_lengths, pad_output_seqs, output_seq_lengths