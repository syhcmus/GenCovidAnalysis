import itertools
import os
import numpy as np
import pandas as pd
from collections import Counter

def encode(seq,encode_map):
    '''
    Encode sequence
    Input: sequence to encode, map to encode
    Output: sequence after encoded
    '''
    final_sequence = []
    for nucleotide in seq:
        if nucleotide in encode_map:
            final_sequence.append(encode_map[nucleotide])

    return final_sequence


def nucleotide_coding(seq,encode_map):
    '''
    Encode nucleotide
    Input: Sequence to encode
    Output: Sequence after encoded
    '''
    
    final_sequence = encode(seq,encode_map)
    _,count = np.unique(final_sequence,return_counts=True)
    nucleotide_0 = count[0]
    nucleotide_1 = count[1]
    encode_seq = ''.join(final_sequence)

    return (encode_seq,nucleotide_0,nucleotide_1)



def get_mean_position(seq,n,c):
    '''
    Get mean position of nucleotide in sequence
    Input: Sequence,number of nucleotide ,nucleotide
    Output: mean position of nucleotide
    '''

    length = len(seq)
    mean = 0

    for idx in range(length):
        if(seq[idx] == c):
            mean += (idx*(1.0/n))

    return mean

def get_variance(seq,n,meu,c):
    '''
    Get variance of nucleotide in sequence
    Input: sequence,number of nucleotide ,nucleotide
    Output: variance of nucleotide
    '''

    length = len(seq)
    variance = 0
    for i in range(length):
        if(seq[i] == c):
            variance += (((i-meu)**2)*1.0)/(n*length)

    return variance

def get_NFV(seq):
    '''
    Calculate novel fast vector of sequence
    Input: sequence to calculate
    Output: novel fast vector
    '''

    R_Y_encode_map = {'A':'R','G':'R','C':'Y','T':'Y'}
    M_K_encode_map = {'A':'M','G':'K','C':'M','T':'K'}
    S_K_encode_map = {'A':'W','G':'S','C':'S','T':'W'}


    (ry_encode_seq,n_r,n_y)  = nucleotide_coding(seq,R_Y_encode_map)
    (mk_encode_seq,n_m,n_k) = nucleotide_coding(seq,M_K_encode_map)
    (sw_encode_seq,n_s,n_w) = nucleotide_coding(seq,S_K_encode_map)

    meu_r = get_mean_position(ry_encode_seq,n_r,'R')
    meu_y = get_mean_position(ry_encode_seq,n_y,'Y')

    meu_m = get_mean_position(mk_encode_seq,n_m,'M')
    meu_k = get_mean_position(mk_encode_seq,n_k,'K')

    meu_s = get_mean_position(sw_encode_seq,n_s,'S')
    meu_w = get_mean_position(sw_encode_seq,n_w,'W')

    D_r = get_variance(ry_encode_seq,n_r,meu_r,'R')
    D_y = get_variance(ry_encode_seq,n_y,meu_y,'Y')

    D_m = get_variance(mk_encode_seq,n_m,meu_m,'M')
    D_k = get_variance(mk_encode_seq,n_k,meu_k,'K')

    D_s = get_variance(sw_encode_seq,n_s,meu_s,'S')
    D_w = get_variance(sw_encode_seq,n_w,meu_w,'W')
    
    Fast_vector = [n_r,meu_r,D_r,n_y,meu_y,D_y,n_m,meu_m,D_m,n_k,meu_k,D_k,n_s,meu_s,D_s,n_w,meu_w,D_w]

    return Fast_vector


def euclidean(list_,seqs_number):
    '''
    Calculate euclidean distance between two sequences
    Input: list of sequence, sequence number
    Output: matrix of sequence
    '''

    matrix = np.zeros([seqs_number, seqs_number])
    for i, j in itertools.combinations(range(0,seqs_number),2):
        matrix[i][j] = matrix [j][i] = np.linalg.norm((list_[i,:]-list_[j,:]),ord=2)
  
    return matrix
   

def create_vects_and_country_wise_distance_matrix():
    '''
    Create country distance vector
    Input: None
    Output: None
    '''

    main_dir = 'data/All_Countries_Splitted'
    files = os.listdir(main_dir)

    if not os.path.exists("data/All_Countries_NFV_Vects"):
        os.mkdir("data/All_Countries_NFV_Vects")
    if not os.path.exists("data/All_Countries_Distance_Matrix"):
        os.mkdir("data/All_Countries_Distance_Matrix")

    for file_ in files:
        inp_file = main_dir + "/"+file_
        c_name = file_.split(".")[0]
        df = pd.read_csv(inp_file)
        
        
        print("Started working with ->",file_)
        sequences = df["Sequence"]
        final_list = []

        for seq in sequences:
            Fast_vector = get_NFV(seq)
            final_list.append(Fast_vector)
    

        acc_vects = np.array(final_list)
        country = np.array(acc_vects)
        
        id_list = df["Accession ID"]
        id_col = pd.Series(id_list)
        vector_col = pd.Series(country.tolist())

        df1 = pd.DataFrame({'Accession ID': id_col,'Vector':vector_col})

        direc_1 = "data/All_Countries_NFV_Vects"
        df1.to_csv(direc_1+"/"+c_name+"_Novel_Fast_Vector.csv",index=False)
        direc_2 = "data/All_Countries_Distance_Matrix"
        matrix = euclidean(country,len(country))

        df2 = pd.DataFrame(matrix,columns=id_list)
        df2.to_csv(direc_2+"/"+c_name+"_distance_matrix.csv",index=False)



if __name__ == "__main__":
    create_vects_and_country_wise_distance_matrix()