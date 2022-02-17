import sklearn.cluster as cluster
import pandas as pd
import numpy as np 
import os
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.metrics import pairwise_distances_argmin_min


method = ""
dataset = ""
direc = ""

def clustering(ACC_ids_list,score_mat,num_seqs,num_clusters):
    '''
    Clustering and assign sequence id for center
    Input: sequence id, matrix distance, number of sequence, number of cluster
    Output: None
    '''
   
    ids_list = ACC_ids_list
   
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = score_mat.index.values
   
    scores = score_mat.to_numpy()

    kmeans = cluster.KMeans(num_clusters)
    results = kmeans.fit(scores)

    cluster_map['cluster'] = results.labels_
    centers = results.cluster_centers_
    closest, _ = pairwise_distances_argmin_min(centers, scores)

    info_handle = open(direc+"/"+dataset+"_center.txt","w")
    info_handle.write(ids_list[closest[0]])
    info_handle.close()



def PCA_on_distance_matrix(matrix):
    '''
    Perform PCA on distance matrix
    Input: Matrix of distance
    Output: Matrix after performing PCA
    '''

    print("Starting PCA")
    PCA_model = PCA(n_components=2)
    scaled_matrix = StandardScaler().fit_transform(matrix)
    PCA_model.fit(scaled_matrix)

    matrix_transformed = PCA_model.transform(scaled_matrix)

    df = pd.DataFrame(matrix_transformed,columns=['PC_1','PC_2'])

    return df

    


def find_representative_sequence():
    '''
    Find representative sequence 
    Input: None
    Output: None
    '''
  
    countries = os.listdir('./data/All_Countries_Distance_Matrix')
    main_direc = './data/All_Countries_Distance_Matrix'
    global direc
    direc = "./data/All_Countries_Representative_Seq"
    print("working with -> "+direc)

    if not os.path.exists(direc):
        os.mkdir(direc)

    for cont_ in countries:
        global dataset,method
        
        dataset = cont_.split("_")[0]
        method = "Represtative_seq"

        csv_file = main_direc +"/"+dataset+"_distance_matrix.csv"

        matrix = pd.read_csv(csv_file)
 
        if(matrix.shape[0] == 1):
            info_handle = open(direc+"/"+dataset+"_center.txt","w")
            info_handle.write(matrix.columns[0])
            info_handle.close()
            continue
        num_clusters = 1
        
        df = PCA_on_distance_matrix(matrix)
        num_seqs = matrix.shape[0]
        clustering(matrix.copy().columns,df,num_seqs,num_clusters)
        

if __name__ == '__main__':
    find_representative_sequence()