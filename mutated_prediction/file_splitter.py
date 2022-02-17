import pandas as pd 
import os
import numpy as np


def split_by_country(df):
    '''
    Extract country name from location
    Input: Dataframe contains location
    Output: List of country name
    '''
    locations = df["Location"]

    loc,_ = np.unique(locations,return_counts=True)

    countries = []
    for country in loc:
        tokens = country.split('/')
        if 'USA' not in country:
            cont_ = country.split('/')[1].strip()
        elif len(tokens) == 2:
            cont_ = country.split('/')[1].strip()
        else:
            cont_ = country.split('/')[2].strip()

        if cont_ == 'New York City':
            cont_ = 'New York'
       
        
        if cont_ not in countries:
            countries.append(cont_)

    countries.sort()

    return countries



def file_spliter():
    '''
    Create file followed by country names
    Input: None
    Output: None
    '''

    label = 'Death'
    required_file = "../input/dataset_labelled_by_" + label +".csv"
    df = pd.read_csv(required_file)
    countries = split_by_country(df.copy())

    if not os.path.exists('data'):
        os.mkdir('data')

    direc = 'data/All_Countries_Splitted'
    if not os.path.exists(direc):
        os.mkdir(direc)

    
    for cont_ in countries:
        n_df = df.loc[df["Location"].str.contains(cont_)]
        f_name = direc+"/"+cont_+".csv"
        n_df.to_csv(f_name,index=False)


if __name__ == "__main__":
    file_spliter()

