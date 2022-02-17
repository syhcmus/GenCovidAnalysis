import itertools
import os
import numpy as np
import pandas as pd
from collections import Counter

def encode(seq,change_map):
    '''
    Encode sequence
    Input: sequence to encode, map to encode
    Output: sequence after encoded
    '''
    n_seq = []
    for base in seq:
        if base in change_map:
            n_seq.append(change_map[base])
    
    assert len(seq) == len(n_seq)

    return n_seq

def R_Y_coding(seq):
    '''
    Encode base to R or Y
    Input: Sequence to encode
    Output: Sequence after encoded
    '''

    change_map = {'A':'R','G':'R','C':'Y','T':'Y'}
    
    n_seq = encode(seq,change_map)
    _,count = np.unique(n_seq,return_counts=True)
    n_r = count[0]
    n_y = count[1]
    ry_encode_seq = ''.join(n_seq)

    return (ry_encode_seq,n_r,n_y)

def M_K_coding(seq):

    '''
    Encode base to M or K
    Input: Sequence to encode
    Output: Sequence after encoded
    '''
    
    change_map = {'A':'M','G':'K','C':'M','T':'K'}

    n_seq = encode(seq,change_map)
    counts = Counter(n_seq)
    n_m = counts['M']
    n_k = counts['K']
    mk_encode_seq = ''.join(n_seq)

    return (mk_encode_seq,n_m,n_k)

def S_W_coding(seq):

    '''
    Encode base to S or W
    Input: Sequence to encode
    Output: Sequence after encoded
    '''

    change_map = {'A':'W','G':'S','C':'S','T':'W'}

    n_seq = encode(seq,change_map)
    counts = Counter(n_seq)
    n_s = counts['S']
    n_w = counts['W']
    sw_encode_seq = ''.join(n_seq)

    return (sw_encode_seq,n_s,n_w)

def get_mean_position(seq,n,c):
    '''
    Get mean position of base in sequence
    Input: Sequence,number of base ,base
    Output: mean position of base
    '''

    length = len(seq)
    mean = 0

    for idx in range(length):
        if(seq[idx] == c):
            mean += (idx*(1.0/n))

    return mean

def get_variance(seq,n,meu,c):
    '''
    Get variance of base in sequence
    Input: sequence,number of base ,base
    Output: variance of base
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

    (ry_encode_seq,n_r,n_y)  = R_Y_coding(seq)
    (mk_encode_seq,n_m,n_k) = M_K_coding(seq)
    (sw_encode_seq,n_s,n_w) = S_W_coding(seq)

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
    assert len(Fast_vector) == 18

    return Fast_vector

def minkowski(list_,seqs_number,exponent): 
    '''
    Calculate minkowski distance between sequence
    Input: list of sequence, sequence number, number of sequence
    Output: matrix of distance
    '''
    matrix = np.zeros([seqs_number, seqs_number])
    for i, j in itertools.combinations(range(0,seqs_number),2):
         matrix[i][j]= matrix [j][i] = np.linalg.norm((list_[i,:]-list_[j,:]),ord=exponent)
  
    return matrix

def euclidean(list_,seqs_number):
    '''
    Calculate euclidean distance between two sequences
    Input: list of sequence, sequence number
    Output: matrix of sequence
    '''
    return minkowski(list_,seqs_number,2)

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
        cont_ = np.array(acc_vects)
        
        ID_arr = df["Accession ID"]
        ID_col = pd.Series(ID_arr)
        Vector_col = pd.Series(cont_.tolist())
        frame = {'Accession ID': ID_col,'Vector':Vector_col}
        df_final = pd.DataFrame(frame)
        direc_1 = "data/All_Countries_NFV_Vects"
        df_final.to_csv(direc_1+"/"+c_name+"_Novel_Fast_Vector.csv",index=False)
        direc_2 = "data/All_Countries_Distance_Matrix"
        matrix = euclidean(cont_,len(cont_))

        final_df = pd.DataFrame(matrix,columns=ID_arr)
        final_df.to_csv(direc_2+"/"+c_name+"_distance_matrix.csv",index=False)



if __name__ == "__main__":
    create_vects_and_country_wise_distance_matrix()