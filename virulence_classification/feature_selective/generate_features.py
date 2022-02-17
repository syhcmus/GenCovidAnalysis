from operator import index
import pandas as pd
import itertools
import numpy as np
import os
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def gap_features(df, nucleotides, max_gap = 30):
    '''
    Calculate the number of occurrence of pair nucleotide with the specific gap
    Input: dataframe, nulcleotides, gap between them
    Output: number of occurence
    '''

    result = pd.DataFrame()
    for i in range(1, max_gap + 1):
        for x1 in nucleotides:
            for x2 in nucleotides:
                col_name = 'GAP_' + x1 + '_' + str(i) + '_' + x2
                result[col_name] = pd.Series(data=(df.shape[0] * [0])).astype(np.int32)
                idx = 0
                for RNA in df['Sequence']:
                    cnt = 0
                    for j in range(len(RNA) - (i + 1)):
                        if RNA[j] == x1 and RNA[j + i + 1] == x2:
                            cnt += 1
                    result[col_name].at[idx] = np.int32(cnt)
                    idx += 1

    return result


def position_independent(df, order, nucleotides, other_nucleotides):
    '''
    Calculate the number of occcurence of k-mer with the length of k
    Input: dataframe, order, nucleotides
    Output: Number of occurence
    '''

    result = pd.DataFrame()

    for ord_ in range(1, order + 1):
        for p in itertools.product(nucleotides, repeat=ord_):
            p = ''.join(p)
            result[p] = pd.Series(data=(df.shape[0] * [0])).astype(np.int32)
            idx = 0
            for RNA in df['Sequence']:
                cnt = RNA.count(p)
                result[p].at[idx] = np.int16(cnt)
                idx += 1

    for p in other_nucleotides:
        result[p] = pd.Series(data=(df.shape[0] * [0])).astype(np.int32)
        idx = 0
        for RNA in df['Sequence']:
            cnt = RNA.count(p)
            result[p].at[idx] = np.int32(cnt)
            idx += 1

    return result


def position_specific(df, order, nucleotides, max_length=30000):
    '''
    Idenfity the occurrence of k-mer with length of k whethe is exist at specific position
    Input: dataframe, order, nucleotides, length of k-mer
    Output: binary values
    '''

    subseq_list = itertools.product(nucleotides, repeat=order)
    count = 0
    subseq_dict = {}
    result = pd.DataFrame()
    num_features = max_length // order


    for p in subseq_list:
        p = ''.join(p)
        count += 1
        subseq_dict[p] = count

    for i in range(num_features):
        p = 'pos_' + str(order*i) + '_' + str(order*(i+1)-1)
        result[p] = pd.Series(data=(df.shape[0] * [0])).astype(np.int32)

    idx = 0
    for RNA in df['Sequence']:
        length = len(RNA)
        for i in range(0, min(length, max_length), order):
            substr = RNA[i:i+order]
            p = 'pos_' + str(i) + '_' + str(i + order - 1)
            if substr in subseq_dict:
                result[p].at[idx] = np.int32(subseq_dict[substr])
        idx += 1


    return result


def generate_features(df, filename=''):
    '''
    Generate features of biological features of gene
    Input: dataset of gen
    Output: features
    '''

    print("generate features")

    dir = 'features'
    if not os.path.exists(dir):
        os.makedirs(dir)

    nucleotides_ = ['A', 'C', 'T', 'G']
    iupac_neucleotides = []

    df['Sequence'] = df['Sequence'].str.upper()

    print('Data Shape: ', df.shape)

    print("Generating features of independent position")
    df_pos_ind = position_independent(df, 4, nucleotides_, iupac_neucleotides).astype(np.int32)


    print("Generating features of specific position")
    df_pos_ps = position_specific(df, 5, nucleotides_).astype(np.int32)


    print("Generating features of gap features")
    df_gap = gap_features(df, nucleotides_).astype(np.int32)

    result = pd.concat([df_pos_ind, df_gap, df_pos_ps], axis=1, sort=False).astype(np.int32)
    result.to_csv(f"{dir}/{filename}",index=False)

    # result = pd.read_csv(f"{dir}/{filename}")

    return result


if __name__ == "__main__":
    train_filename = 'train'
    generate_features(train_filename)
    
