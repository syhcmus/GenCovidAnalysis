import sklearn.cluster as cluster
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min


method = ""
dataset = ""
direc = ""

def clustering(ids_list,score_matrix,num_clusters):
    '''
    Clustering and assign sequence id for center
    Input: sequence id, matrix distance, number of sequence, number of cluster
    Output: None
    '''
   
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = score_matrix.index.values
   
    scores = score_matrix.to_numpy()

    kmeans = cluster.KMeans(num_clusters)
    results = kmeans.fit(scores)

    cluster_map['cluster'] = results.labels_
    centers = results.cluster_centers_
    closest, _ = pairwise_distances_argmin_min(centers, scores)

    clusters = open(direc+"/"+dataset+"_center.txt","w")
    clusters.write(ids_list[closest[0]])
    clusters.close()



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

    for country in countries:
        global dataset,method
        
        dataset = country.split("_")[0]
        method = "Represtative_seq"

        csv_file = main_direc +"/"+dataset+"_distance_matrix.csv"

        matrix = pd.read_csv(csv_file)
 
        if(matrix.shape[0] == 1):
            clusters = open(direc+"/"+dataset+"_center.txt","w")
            clusters.write(matrix.columns[0])
            clusters.close()
            continue
            
        num_clusters = 1
        
        df = PCA_on_distance_matrix(matrix)
        clustering(matrix.copy().columns,df,num_clusters)
        

if __name__ == '__main__':
    find_representative_sequence()