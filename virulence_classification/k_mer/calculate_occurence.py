import numpy as np
import itertools



def get_motifs_V2(length):

    (d,index) = ({}, 0)
    lst = ['A','C','G','T']
    els =  list(itertools.product(lst,repeat = length))
    els = [''.join(tups) for tups in els]
    for elems in els:
        d[elems] = index
        index+=1

    return d

def calculate_occurrences(input_file, length=7):
    print("calculating occurence ...")
    sequences_list = input_file["Sequence"]
    d = get_motifs_V2(length)   
    rows_num = len(sequences_list)
    cols_num = len(d)
    #with open("k-mers.txt","w") as out_file:
     #   pprint(d, stream=out_file)

    data = np.zeros(shape=(rows_num,cols_num))
    y_indicator = np.zeros(shape=(rows_num,1))
    for row_idx, seq in enumerate(sequences_list):
        for i in range(0, len(seq)-length+1):
            word = seq[i:i+length]
            if word in d:
                col_idx = d[word]
                data[row_idx, col_idx] += 1
        y_indicator[row_idx] = input_file["Indicator"][row_idx]

    return (data,y_indicator)