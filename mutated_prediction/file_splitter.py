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
            country = country.split('/')[1].strip()
        elif len(tokens) == 2:
            country = country.split('/')[1].strip()
        else:
            country = country.split('/')[2].strip()

        if country == 'New York City':
            country = 'New York'
       
        
        if country not in countries:
            countries.append(country)

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

    
    for country in countries:
        countries_df = df.loc[df["Location"].str.contains(country)]
        f_name = direc+"/"+country+".csv"
        countries_df.to_csv(f_name,index=False)


if __name__ == "__main__":
    file_spliter()

