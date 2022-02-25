import pandas as pd
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
import random
import os

def read_fasta(input_fasta):
    '''
    Read sequences from the fasta files
    Input: fasta files
    Output: dictionary
    '''

    sequence_map = {}
    for sequence_record in SeqIO.parse(input_fasta,"fasta"):
        sequence_id = sequence_record.description.split("|")[1]
        sequence_map[sequence_id] = str(sequence_record.seq)

    return sequence_map

def processing(label_name_file,original_df,Label):
    '''
    Creates dataframe supplied with labels
    Input: datataset labelling helper file
    Output: dataframe 
    '''

    label_df = pd.read_csv(label_name_file)
    original_df = original_df.dropna()

    locations = original_df["Location"]
    countries = label_df["Countries"]
    deaths = label_df["Death"]

    indicator_col = f"Indicator_{Label}"
    indicator = label_df[indicator_col]

    death_arr = np.full(locations.shape[0],-1)
    indicator_arr = np.full(locations.shape[0],-1)


    for location_index,location_value in enumerate(locations):
        location_value = str(location_value).split("/")
        for country_index,country_value in enumerate(countries):
            if(country_value in str(location_value).encode('ascii', 'ignore').decode('ascii')):
                death_arr[location_index] = deaths[country_index]
                indicator_arr[location_index] = indicator[country_index]

    original_df["Death"] = death_arr
    original_df["Indicator"] = indicator_arr

    permitted_label = [0,1]
    final_df = original_df.loc[original_df["Indicator"].isin(permitted_label)]

    file_name = f"./input/dataset_labelled_by_{Label}.csv"
    final_df.to_csv(file_name,index=False)

    return final_df


def change_sequence(sequence):
    '''
    Helper function for change sequences
    Input: sequence
    Output: sequence after changing nucleotide     
    '''

    
    sequence_map = {
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
    keys = sequence_map.keys()

    
    seq = sequence.upper()
    seq = seq.replace("-","")
    
    mutate_seq = [c for c in seq]
    for i in range(len(seq)):
        if(seq[i] in keys):
            mutate_seq[i] = random.choice(sequence_map[seq[i]])

    unique_nucleotide = np.unique(mutate_seq)
    permitted_list = ['A','C','G','T']
    if(len(unique_nucleotide) != 4):
        omitted_nucleotide = [nucleotide for nucleotide in unique_nucleotide if nucleotide not in permitted_list]
        for nucleotide in omitted_nucleotide:
            mutate_seq = list(filter((nucleotide).__ne__, mutate_seq))
    
    final_sequence = ''.join(mutate_seq)

    return final_sequence

def change_and_check_sequences(sequences):
    '''
    Interpret the sequences and the check it
    Input: sequence
    Output: sequence after changing
    '''
    

    final_sequence = []
    for seq in sequences:

        s = set(seq)
        if len(s) != 4:
            n_seq = change_sequence(seq)
            final_sequence.append(n_seq)
        else:
            final_sequence.append(seq)

   
    return final_sequence

def data_preparation(sequence_map,info_file,label_method,label_helper_file):
    '''
    Creates dataset labeled by the label method provided
    Input: dictionary, the info_file name and the label name
    Output: None
    '''

    id_list = list(sequence_map.keys())
    cols = ["Accession ID","Virus name","Location","Collection date"]
    info_df = pd.read_csv(info_file)
    info_df = info_df.loc[info_df["Accession ID"].isin(id_list)][cols]
    info_df['Sequence'] = info_df['Accession ID'].map(sequence_map) 
    sequences = np.array(info_df['Sequence'])
    info_df['Sequence'] = change_and_check_sequences(sequences.copy()) 
    

    df =  processing(label_helper_file,info_df,label_method)
    y = df["Indicator"]

    train, test = train_test_split(df, test_size=0.2,stratify=y, random_state=0)
    train.to_csv("./input/train.csv", index=False)
    test.to_csv("./input/test.csv", index=False)

 

def main():
    input_fasta = './data/sequence.fasta'
    info_file = './data/covid_virus_acknowledgement.csv'
    label_helper_file = './data/dataset_labelling_helper.csv'
    label = 'Death'

    print("Preprocessing data ...") 

    if not os.path.exists('input'):
        os.mkdir('input')

    sequence_map = read_fasta(input_fasta)
    data_preparation(sequence_map,info_file,label,label_helper_file)

    

if __name__ == "__main__":
    main()