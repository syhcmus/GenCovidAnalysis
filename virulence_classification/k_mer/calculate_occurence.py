import numpy as np
import itertools

def get_k_mers(length):

    '''
    Get the k-mer set with the specific length
    Input: length of k-mer
    Output: set of k-mer
    '''
   
    k_mers_map = {}
    col_index = 0

    nucleotide = ['A','C','G','T']
    k_mers =  list(itertools.product(nucleotide,repeat = length))
    k_mers = [''.join(tups) for tups in k_mers]
    for k_mer in k_mers:
        k_mers_map[k_mer] = col_index
        col_index += 1

    return k_mers_map

def calculate_occurrences(df, length=7):
    '''
    Calulate occurence of k-mers set with the specify length
    Input: data containe sequence, length of k-mer
    Output: occuence of k-mers set
    '''

    print("calculating occurence ...")
    sequences_list = df["Sequence"]
    k_mers_map = get_k_mers(length)   
    rows_num = len(sequences_list)
    cols_num = len(k_mers_map)

    features = np.zeros(shape=(rows_num,cols_num))
    y = np.zeros(shape=(rows_num,1))

    for row_idx, seq in enumerate(sequences_list):
        for i in range(0, len(seq)-length+1):
            k_mer = seq[i:i+length]
            if k_mer in k_mers_map:
                col_idx = k_mers_map[k_mer]
                features[row_idx, col_idx] += 1
                
        y[row_idx] = df["Indicator"][row_idx]

    return (features,y)