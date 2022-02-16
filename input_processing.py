from operator import index
import pandas as pd
import numpy as np
from Bio import SeqIO
import argparse
from sklearn.model_selection import train_test_split
import random
old_flag = False

def read_fasta(input_fasta):
    '''
    Read sequences from the fasta files
    Input: fasta files
    Output: dictionary
    '''

    id_seq_map = {}
    for seq_record in SeqIO.parse(input_fasta,"fasta"):
        id_ = seq_record.description.split("|")[1]
        id_seq_map[id_] = str(seq_record.seq)

    return id_seq_map

def processing(label_name_file,original_df,Label):
    '''
    Creates dataframe supplied with labels
    Input: datataset labelling helper file
    Output: dataframe 
    '''

    # label_helper_df = pd.read_csv("Dataset_Labelling_Helper.csv")
    label_helper_df = pd.read_csv(label_name_file)
    original_df = original_df.dropna()

    locations = original_df["Location"]
    countries = label_helper_df["Countries"]
    deaths = label_helper_df["Death"]

    ind_ = "Indicator_" + Label
    indicator = label_helper_df[ind_]

    death_arr = np.full(locations.shape[0],-1)
    indicator_arr = np.full(locations.shape[0],-1)


    for l_idx,l_value in enumerate(locations):
        l_value = str(l_value).split("/")
        for c_idx,c_value in enumerate(countries):

            if(c_value in str(l_value).encode('ascii', 'ignore').decode('ascii')):
                death_arr[l_idx] = deaths[c_idx]
                indicator_arr[l_idx] = indicator[c_idx]

    original_df["Death"] = death_arr
    original_df["Indicator"] = indicator_arr

    permitted_label = [0,1]
    good_df = original_df.loc[original_df["Indicator"].isin(permitted_label)]

    file_name = f"./input/dataset_labelled_by_{Label}.csv"
    good_df.to_csv(file_name,index=False)

    return good_df


def change_seq(sequence):
    '''
    Helper function for change sequences
    Input: sequence
    Output: sequence after changing nucleotide     
    '''

    
    change_map = {
        "R":["A","G"],
        "Y":["C","T"],
        "S":["C","G"],
        "W":["A","T"],
        "K":["G","T"],
        "M":["A","C"],
        "B":["C","T","G"],
        "D":["A","T","G"],
        "H":["C","T","A"],
        "V":["C","A","G"],
        "N":["A","C","T","G"]
    }
    keys = change_map.keys()

    # change_map["R"] = ["A","G"]
    # change_map["Y"] = ["C","T"]
    # change_map["S"] = ["C","G"]
    # change_map["W"] = ["A","T"]
    # change_map["K"] = ["G","T"]
    # change_map["M"] = ["A","C"]
    # change_map["B"] = ["C","T","G"]
    # change_map["D"] = ["A","T","G"]
    # change_map["H"] = ["C","T","A"]    
    # change_map["V"] = ["C","A","G"]
    # change_map["N"] = ["A","C","T","G"]
    # keys = change_map.keys()
    

    seq = sequence.upper()
    seq = seq.replace("-","")
    
    mutate_seq = [c for c in seq]
    for i in range(len(seq)):
        if(seq[i] in keys):
            mutate_seq[i] = random.choice(change_map[seq[i]])

    unique = np.unique(mutate_seq)
    permitted_list = ['A','C','G','T']
    if(len(unique) != 4):
        nucleotide_omitted = [x for x in unique if x not in permitted_list]
        for one_by_one in nucleotide_omitted:
            mutate_seq = list(filter((one_by_one).__ne__, mutate_seq))
    

    n_sequence = ''.join(mutate_seq)

    return n_sequence

def change_and_check_sequences(sequences):
    '''
    Interpret the sequences and the check it
    Input: sequence
    Output: sequence after changing
    '''
    

    n_sequence = []
    for seq in sequences:

        s = set(seq)
        if(len(s)!=4):
            n_seq = change_seq(seq)
            n_sequence.append(n_seq)
        else:
            n_sequence.append(seq)

    for seq in n_sequence:
        s = set(seq)
   
    return n_sequence

def data_preparation(id_seq_map,info_file,label_method,label_helper_file):
    '''
    Creates dataset labeled by the label method provided
    Input: dictionary, the info_file name and the label name
    Output: None
    '''

    cols = ["Accession ID","Virus name","Location","Collection date"]
    info_df = pd.read_csv(info_file)

    id_list = list(id_seq_map.keys())
    usable_info_df = info_df.loc[info_df["Accession ID"].isin(id_list)]
    usable_info_df = usable_info_df[cols]

    usable_info_df['Sequence'] = usable_info_df['Accession ID'].map(id_seq_map) # Embedding Sequences in the csv file
    sequences = np.array(usable_info_df['Sequence'])
    usable_info_df['Sequence'] = change_and_check_sequences(sequences.copy()) # Interpreting the symbols
    

    all_df =  processing(label_helper_file,usable_info_df,label_method)
    y = all_df["Indicator"]

    train, test = train_test_split(all_df, test_size=0.2,stratify=y, random_state=0)
    train.to_csv("./input/train.csv", index=False)
    test.to_csv("./input/test.csv", index=False)

 

def main():
    input_fasta = './data/sequence.fasta'
    info_file = './data/covid_virus_acknowledgement.csv'
    label_helper_file = './data/dataset_labelling_helper.csv'
    label = 'Death'

    print("Preprocessing data ...") 

    id_seq_map = read_fasta(input_fasta)
    data_preparation(id_seq_map,info_file,label,label_helper_file)

    print("Done") 
    

if __name__ == "__main__":
    main()