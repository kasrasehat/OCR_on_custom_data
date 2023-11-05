import pickle
import numpy as np
import torch


def read_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        return f"An error occurred: {e}"


max_length = 160
tokens = read_pickle_file('E:/codes_py/Larkimas/tokens_list.pkl')
tokens[:0] = ['<sos>', '<eos>', 'a', 'b', 'c', 'd']
tokens.append('<pad>')

def char2ind(char):
    return tokens.index(char)


def ind2char(ind):
    return tokens[ind]


def ind2one_hot(ind):
    one_hot = torch.zeros((1, len(tokens)))
    one_hot[0, ind] = 1
    return one_hot


def tensor_from_str(str):
    n = len(str)
    encode = torch.zeros((max_length, len(tokens)))
    for i, char in enumerate(str):
        encode[i, :] = ind2one_hot(char2ind(char))
    return encode

