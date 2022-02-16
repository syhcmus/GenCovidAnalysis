from operator import mod
from keras.models import Sequential,load_model
from keras.layers import LSTM,Conv1D,MaxPooling1D,Activation,Flatten,Bidirectional,Embedding,Masking,Dense,Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from tensorflow.keras.optimizers import RMSprop
import pandas as pd 
import numpy as np 
import collections
import os
from random import randrange
from numpy import save


from keras.preprocessing.text import Tokenizer

date_country_map = {}
final_input_map = {}


INPUT_FOLDER = "data/"
OUTPUT_FOLDER = "results/"



N_EPOCHS = 1


direc = INPUT_FOLDER+"All_Countries_Representative_Seq"
Fast_Vector_direc = INPUT_FOLDER+"All_Countries_NFV_Vects"
Other_Info_File = './../input/dataset_labelled_by_Death.csv'

all_cluster_direc = INPUT_FOLDER+"All_Clusters"


class Seq_Wrapper:
    def __init__(self,seq,use_indicator=0):
        self.seq = seq
        self.indicator = use_indicator
    def __repr__(self):
        return "<Seq:%s, Indicator:%d>" % (self.seq, self.indicator)

    def __str__(self):
        return "<Seq:%s, Indicator:%d>" % (self.seq, self.indicator)
    
    def get_indicator(self):
        return self.indicator
    
    def get_seq(self):
        return self.seq

    def set_indi(self,indicator):
      self.indicator = indicator

def formatter(date_):
    return date_.strftime("%d_%m_%Y")

def info_sequencer(cluster_file,num,start,end):
    global date_country_map

    print("Working with cluster", cluster_file)
    c_fl = open(cluster_file,"r")
    id_list = c_fl.read().split("\n")
    # Fast_Vector_File = Fast_Vector_direc + "/" + num + "_Novel_Fast_Vector.csv"
    # Fast_vector_df = pd.read_csv(Fast_Vector_File)
    Other_Info_df = pd.read_csv(Other_Info_File)
    # final_df_1 = Fast_vector_df.loc[Fast_vector_df["Accession ID"].isin(id_list)]
    final_df = Other_Info_df.loc[Other_Info_df["Accession ID"].isin(id_list)]

    # final_df = pd.merge(final_df_1, final_df_2, on='Accession ID')
    final_cols = ["Accession ID","Virus name","Location","Collection date","Death","Sequence"]
    final_df = final_df[final_cols]

    direc_point = None
    for point,maps in date_country_map.items():
        if(num in maps):
            direc_point = formatter(point)
            # sequences = final_df["Sequence"]
            final_input_map[point][num] = [Seq_Wrapper(t_seq[start:end]) for t_seq in final_df["Sequence"]]
    # twenty_seq_df.append(final_df)
    if direc_point != None:
        final_df_name = num+ "_representative_seq" +".csv"
        final_df.to_csv(all_cluster_direc +"/"+direc_point+"/"+final_df_name)



def get_cluster_dist(cluster1,cluster2):
    vect1 = cluster1["Fast Vector"]
    # print(vect1[0])
    vect2 = cluster2["Fast Vector"]
    # print(vect2.shape)
    final_vect = np.zeros([vect1.shape[0],vect2.shape[0]])
    # print(final_vect.shape)
    for i in range(vect1.shape[0]):
        for j in range(vect2.shape[0]):
            var_1 = np.asarray(vect1[i].replace("[","").replace("]","").replace(' ','').split(",")).astype(np.float)
            var_2 = np.asarray(vect2[j].replace("[","").replace("]","").replace(' ','').split(",")).astype(np.float)

            final_vect[i][j] = np.linalg.norm((var_1-var_2),ord=2)
    dist = np.sum(final_vect)

    return dist

def file_name_checker(cluster_file):
    if(cluster_file.startswith("USA")):
        bleh = cluster_file.split("_")
        num = bleh[0] + "_" + bleh[1]
    else:
        num = cluster_file.split("_")[0]
    return num

def file_array_formatter(file_arr):
    return [file_name_checker(fl) for fl in file_arr]


    
def CNN(dim1,dim2,output_dim):

    model = Sequential()
    model.add(Conv1D(filters=1024,
                        kernel_size=24,
                        trainable=True,
                        padding='valid',
                        activation='relu',
                        strides=1,input_shape=(dim1,dim2)))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(256,return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=["accuracy"])

    return model


def RNN(dim,lstm_cells=64):
    """Make a word level recurrent neural network with option for pretrained embeddings
       and varying numbers of LSTM cell layers."""

    model = Sequential()

    # Map words to an embedding
    
    model.add(
        Embedding(
            input_dim=dim,
            output_dim=50,
            weights=None,
            trainable=False,
            mask_zero=True))
    model.add(Masking())


    model.add(
        LSTM(
            lstm_cells,
            return_sequences=False,
            dropout=0.1,
            recurrent_dropout=0.1))
        
    model.add(Dense(64, activation='relu'))
    # Dropout for regularization
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(dim, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model



def make_single_position_dataset(pos):
    ####### run at the time of loading the time series sequences #############
    with open(OUTPUT_FOLDER+'all_time_series.txt', 'r') as f:
        list_time_series_all = f.read().splitlines()


    ######### mapping chars to integers ##############
    chars = "ATGC"
    char_to_int_for_y = dict((c, i) for i, c in enumerate(chars))
 

    ################## taking full sequence ###################
    n_chars = len(list_time_series_all[0])
    step = 1

    input_for_lstm = []
    output_for_lstm = []

    for time_series in list_time_series_all:   
        dataX = []

        for i in range(0, n_chars-pos, step):
            seq_in = time_series[i]
            dataX.append(char_to_int_for_y[seq_in])

        seq_out = time_series[n_chars-pos]

        one_output = char_to_int_for_y[seq_out]
    #     one_output = to_categorical(one_output)
        output_for_lstm.append(one_output)
        
        # dataX = np_utils.to_categorical(dataX)
        input_for_lstm.append(dataX)


    input_for_lstm = np.array(input_for_lstm)
    output_for_lstm = np.array(output_for_lstm)

    output_for_lstm = np_utils.to_categorical(output_for_lstm)

    input_for_lstm = np_utils.to_categorical(input_for_lstm)

    X_train, X_test, y_train, y_test = train_test_split(
    input_for_lstm, output_for_lstm, test_size=0.20, random_state=42)


    ###### saving numpy arrays ###########3
    save(OUTPUT_FOLDER+'X_train.npy', X_train)
    save(OUTPUT_FOLDER+'X_test.npy', X_test)
    save(OUTPUT_FOLDER+'y_train.npy', y_train)
    save(OUTPUT_FOLDER+'y_test.npy', y_test)


def make_sequence(texts,
                   lower=False,
                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    """Turn a set of texts into sequences of integers"""

    # Create the tokenizer object and train on texts
    tokenizer = Tokenizer(lower=lower, filters=filters)
    tokenizer.fit_on_texts(texts)

    # Create look-up dictionaries and reverse look-ups
    word_idx = tokenizer.word_index
    num_words = len(word_idx) + 1


    print(f'There are {num_words} unique words.')

    # Convert text to sequences of integers
    sequences = tokenizer.texts_to_sequences(texts)


    training_seq = []
    labels = []

    # Iterate through the sequences of tokens
    for seq in sequences:
        # Set the features and label
        training_seq.append(seq[:-1])
        labels.append(seq[-1])

    return  num_words, training_seq, labels


from sklearn.utils import shuffle

def make_train_test(features,
                       labels,
                       num_words,
                       train_fraction=0.8):
    """Create training and validation features and labels."""

    # Randomly shuffle features and labels
    features, labels = shuffle(features, labels, random_state=42)

    # Decide on number of samples for training
    train_end = int(train_fraction * len(labels))

    train_features = np.array(features[:train_end])
    valid_features = np.array(features[train_end:])

    train_labels = labels[:train_end]
    valid_labels = labels[train_end:]

    # Convert to arrays
    X_train, X_valid = np.array(train_features), np.array(valid_features)

    # Using int8 for memory savings
    y_train = np.zeros((len(train_labels), num_words), dtype=np.int8)
    y_valid = np.zeros((len(valid_labels), num_words), dtype=np.int8)

    # One hot encoding of labels
    for example_index, word_index in enumerate(train_labels):
        y_train[example_index, word_index] = 1

    for example_index, word_index in enumerate(valid_labels):
        y_valid[example_index, word_index] = 1


    return X_train, X_valid, y_train, y_valid


def make_sequence_dataset():
    time_series_all = None
    with open(OUTPUT_FOLDER+'all_time_series.txt', 'r') as f:
        time_series_all = f.read().splitlines()


    filters = '!"#$%&()*+/:<=>@[\\]^_`{|}~\t\n'
    num_words,features, labels = make_sequence(time_series_all, filters=filters)

    X_train, X_test, y_train, y_test = make_train_test(
    features, labels, num_words)    
    
    ###### saving numpy arrays ###########3
    save(OUTPUT_FOLDER+'X_train.npy', X_train)
    save(OUTPUT_FOLDER+'X_test.npy', X_test)
    save(OUTPUT_FOLDER+'y_train.npy', y_train)
    save(OUTPUT_FOLDER+'y_test.npy', y_test)



def predict(model_name, single_position):
    ######### loading test dataset #############
    X_test = np.load(OUTPUT_FOLDER+'X_test.npy')
    y_test = np.load(OUTPUT_FOLDER+'y_test.npy')


    model = None

    if single_position and model_name == 'CNN':
        model = load_model(OUTPUT_FOLDER+'single_position_CNN.h5')
    elif model_name == 'CNN':
        X_test = np_utils.to_categorical(X_test)
        model = load_model(OUTPUT_FOLDER+'sequence_CNN.h5')
    else:
        model = load_model(OUTPUT_FOLDER+'sequence_RNN.h5')

    eval = model.evaluate(X_test, y_test, verbose=1)


    return eval



def train(model_name, single_position):


    ######### loading train dataset #############
    X_train = np.load(OUTPUT_FOLDER+'X_train.npy')
    y_train = np.load(OUTPUT_FOLDER+'y_train.npy')
    


    model = None

    if single_position and model_name == 'CNN':

        model = CNN(X_train.shape[1], X_train.shape[2], y_train.shape[1])
        model.summary()


        callbacks_list = [
            ModelCheckpoint(
                filepath='single_position_CNN_cb.h5',
                monitor='val_loss', save_best_only=True),
        ]

        model.fit(X_train,
                    y_train,
                    batch_size=64,
                    epochs=N_EPOCHS,
                    callbacks=callbacks_list,
                    validation_split=0.15,
                    verbose=1)



        model.save(OUTPUT_FOLDER+'single_position_CNN.h5')
    
    elif model_name == 'CNN':
        X_train = np_utils.to_categorical(X_train)

        model = CNN(X_train.shape[1], X_train.shape[2], y_train.shape[1])
        model.summary()


        callbacks_list = [
            ModelCheckpoint(
                filepath='sequence_CNN_cb.h5',
                monitor='val_loss', save_best_only=True),
        ]

        model.fit(X_train,
                    y_train,
                    batch_size=2048,
                    epochs=N_EPOCHS,
                    callbacks=callbacks_list,
                    validation_split=0.15,
                    verbose=1)


        model.save(OUTPUT_FOLDER+'sequence_CNN.h5')

    else:

        model = RNN(y_train.shape[1])
        model.summary()


        callbacks_list = [
            ModelCheckpoint(
                filepath='sequence_RNN_cb.h5',
                monitor='val_loss', save_best_only=True),
        ]

        model.fit(X_train,
                    y_train,
                    batch_size=2048,
                    epochs=N_EPOCHS,
                    callbacks=callbacks_list,
                    validation_split=0.15,
                    verbose=1)


        model.save(OUTPUT_FOLDER+'sequence_RNN.h5')


    
    # model1.load(OUTPUT_FOLDER+'CNN_BI_LSTM_pos_'+str(POS_OF_MUTATION)+'_val_15.h5')




def prepare_data(start,end,single_position):
    global date_country_map
    global final_input_map

    end = end + 1

    if not os.path.exists(all_cluster_direc):
        os.mkdir(all_cluster_direc)

    df_1 = pd.read_csv('./../data/first_case_occurrence.csv')
    df_1["First_Case"] = pd.to_datetime(df_1["First_Case"])
    date_ = df_1["First_Case"]

    
    for date in date_:
        
        same_date_df = df_1.loc[df_1["First_Case"] == date]

        if date.date() not in date_country_map:
            date_country_map[date.date()] = same_date_df["Country"].values

    dub = date_country_map
    date_country_map = collections.OrderedDict(sorted(dub.items()))


    Other_df = pd.read_csv(Other_Info_File)
    for point in date_country_map:
        country_list = date_country_map[point]
        final_input_map[point] = {}
        
        for country in country_list:
            if(country == "USA"):
                continue
            s_name = country
            d_name = country
            if(country == "Korea" or country == "South Korea"):
                d_name = "South Korea"
            if(country == "England" or country == "United Kingdom"):
                d_name = "United Kingdom"
            
            temp_ = Other_df.loc[Other_df["Location"].str.contains(s_name)]

            if(len(temp_) == 0):
                continue
            
        
            if(d_name not in final_input_map[point]):
                final_input_map[point][d_name] = True   


    for point in final_input_map:
        if(len(final_input_map[point]) == 0):
            del date_country_map[point] 
        


    for key in date_country_map:
        key = key.strftime("%d_%m_%Y")
        if not os.path.exists(all_cluster_direc+"/"+key):
            os.mkdir(all_cluster_direc+"/"+key)

    def formatter(date_):
        return date_.strftime("%d_%m_%Y")

    final_input_map = {}
    for point in date_country_map:
        final_input_map[point] = {}


    file_list = os.listdir(direc)

    for cluster_file in file_list:
        if(cluster_file.endswith("txt")):
            if(cluster_file.startswith("USA")):
                bleh = cluster_file.split("_")
                num = bleh[0] + "_" + bleh[1]
            else:
                num = cluster_file.split("_")[0]

            info_sequencer(direc+"/"+cluster_file,num,start,end)


    for point in final_input_map:
        for small_clust in final_input_map[point]:
            print(point,small_clust,len(final_input_map[point][small_clust]))

    dub = final_input_map
    final_input_map = collections.OrderedDict(sorted(dub.items()))


    keys = list(date_country_map.keys())
    edge_map = {}
    for idx in range(len(date_country_map)-1):
        point_i = keys[idx]
        point_j = keys[idx+1]
        
        file_list_i = os.listdir(all_cluster_direc+"/"+formatter(point_i))
        file_list_j = os.listdir(all_cluster_direc+"/"+formatter(point_j))

        val_i = file_array_formatter(file_list_i)
        val_j = file_array_formatter(file_list_j)
        edge_map[point_i] = {}

        if(len(val_i) == 0 or len(val_j) == 0):
            continue
        for val in val_i:
            edge_map[point_i][val] = val_j

    total_time_series = 300000
    time_series_all = {}
    total_unique_seq = 0

    while True:
        one_time_series = ""
        one_time_series1 = []
        current_country = "China"

        for date in final_input_map.keys():

            total_seq = len(final_input_map[date][current_country])
            seq_idx = -1
            for i in range(total_seq):
                if final_input_map[date][current_country][i].get_indicator() == 0:
                    seq_idx = i
                    ###### set indicator 1
                    final_input_map[date][current_country][i].set_indi(1)
                    break
            if seq_idx == -1:
                seq_idx = randrange(total_seq) 

            sequence = final_input_map[date][current_country][seq_idx].get_seq()

            one_time_series = one_time_series + sequence
            one_time_series1.append(sequence)
                        
            # find next country from edge_map to select seq from
            if date in edge_map.keys():        
                total_next_country = len(edge_map[date][current_country])
                next_country_idx = randrange(total_next_country)
                current_country = edge_map[date][current_country][next_country_idx]


        if single_position == False :
            one_time_series = ' '.join(one_time_series1)

        if not (one_time_series in time_series_all.keys()):
            time_series_all[one_time_series] = 1
            total_unique_seq = total_unique_seq + 1
            print(total_unique_seq)
     
        if total_unique_seq == total_time_series:
            print("breaking------")
            break


    list_time_series_all = list(time_series_all.keys())

    if not os.path.exists('results'):
        os.mkdir('results')

    with open(OUTPUT_FOLDER+'all_time_series.txt', 'w') as f:
        for item in list_time_series_all:
            f.write("%s\n" % item)


def transform_data(start,end,pos,single_position):
    prepare_data(start,end,single_position)

    if single_position:
        make_single_position_dataset(pos)
    else: 
        make_sequence_dataset()

    

def main():

    start = 23825 # start site of interests
    end = 23829 # end site of interests
    pos = 3 # single position of genome sequence to predict
    model_name = 'CNN'
    single_position = False

    # prepare_data(start,end,single_position)

    # if single_position:
    #     make_single_position_dataset(pos)
    # else: 
    #     make_sequence_dataset()
    

    train(model_name,single_position)

    eval = predict(model_name,single_position)

    test_accuracy = eval[1]
    print(f'Accuracy: {round(100 * test_accuracy, 2)}%')

if __name__ == '__main__':
    main()