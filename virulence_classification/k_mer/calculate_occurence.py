import numpy as np
import itertools

def get_k_mers(length):

    '''
    Get the k-mer set with the specific length
    Input: length of k-mer
    Output: set of k-mer
    '''
    (dict_,col_index) = ({}, 0)

    base = ['A','C','G','T']
    elements =  list(itertools.product(base,repeat = length))
    elements = [''.join(tups) for tups in elements]
    for e in elements:
        dict_[e] = col_index
        col_index += 1

    return dict_

def calculate_occurrences(input_file, length=7):
    '''
    Calulate occurence of k-mers set with the specify length
    Input: data containe sequence, length of k-mer
    Output: occuence of k-mers set
    '''

    print("calculating occurence ...")
    sequences_list = input_file["Sequence"]
    dict_ = get_k_mers(length)   
    rows_num = len(sequences_list)
    cols_num = len(dict_)

    data = np.zeros(shape=(rows_num,cols_num))
    y_indicator = np.zeros(shape=(rows_num,1))

    for row_idx, seq in enumerate(sequences_list):
        for i in range(0, len(seq)-length+1):
            word = seq[i:i+length]
            if word in dict_:
                col_idx = dict_[word]
                data[row_idx, col_idx] += 1
                
        y_indicator[row_idx] = input_file["Indicator"][row_idx]

    return (data,y_indicator)