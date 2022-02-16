
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Flatten,Input,LeakyReLU,BatchNormalization,Reshape,Dropout,Activation,Conv1D,MaxPool1D,concatenate
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import numpy as np
from sklearn import metrics
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle as pkl


######### global defination ###############

LOSS_FN = 'binary_crossentropy'  
LEARNING_RATE = 0.001  
n_epochs = 100
BATCH_SIZE = 64



def CNN(dim1,dim2):

    '''
    Create CNN model
    Input: input shape of model
    Output: CNN model
    
    '''

    model = Sequential()
    model.add(Conv1D(filters=10, kernel_size=5, strides=1, padding="same",activation='relu', input_shape=(dim1,dim2)))

    model.add(MaxPool1D(pool_size=2))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Conv1D(filters=20, kernel_size=5, strides=1, padding="same",activation='relu'))

    model.add(MaxPool1D(pool_size=2))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Conv1D(filters=30, kernel_size=5))
    model.add(MaxPool1D(pool_size=2))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Flatten())
    
    model.add(Dense(500))
    model.add(Dropout(0.5))

    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss=LOSS_FN, optimizer=Adam(lr=LEARNING_RATE, amsgrad=True), metrics=['accuracy'])
    model.summary()

    return model


def LGBM():
    '''
    Create LGBM model
    Input: None
    Output: LGBM model
    '''

    clf = LGBMClassifier(objective='binary')

    grid_params = {
    'learning_rate': [ 0.01, 0.05],
    'num_leaves': [50,100],
    'max_depth' : [5,10,15],
    'min_data_in_leaf':[5, 10]
    }

    model = RandomizedSearchCV(clf,grid_params,verbose=1,cv=10,n_iter=10)

    return model


def train(train_x, train_y, model_name=""):
    '''
    Create train model
    Input: train features, train label, model name
    Output: train model
    '''


    if model_name == 'CNN':

        train_x = np.reshape(train_x,(train_x.shape[0],train_x.shape[1],1))

        model = CNN(train_x.shape[1],train_x.shape[2])

        callbacks_list = [
        ModelCheckpoint(
            filepath=f"{model_name}_cb.h5",
            monitor='val_loss', save_best_only=True)
        ]
        
        model.fit(train_x,
                    train_y,
                    batch_size=BATCH_SIZE,
                    epochs=n_epochs,
                    callbacks=callbacks_list,
                    validation_split=0.15,
                    verbose=1)


        model.save_weights(f'{model_name}.h5')

    elif model_name == 'LGBM':
        model = LGBM()
        model.fit(train_x, train_y)

        f = open(f'{model_name}.pkl','wb')
        pkl.dump(model, f)



def score(test_y, pred_y):

    '''
    Evaluate model
    Input: true label, predicted label
    Output: score of model
    '''

    accuracy = metrics.accuracy_score(test_y, pred_y)
    precision = metrics.precision_score(test_y, pred_y)
    recall = metrics.recall_score(test_y, pred_y)
    f1 = metrics.f1_score(test_y, pred_y)

    return accuracy, precision, recall, f1


def predict(test_x, test_y, model_name=''):

    '''
    Use training model to predict
    Input: test features, test label, model name
    Output: score of model
    '''

    prob_y = None
    pred_y = None

    if model_name == 'CNN':
        test_x = np.reshape(test_x,(test_x.shape[0],test_x.shape[1],1))

        model = CNN(test_x.shape[1],test_x.shape[2])
        model.load_weights(f'{model_name}.h5')

        prob_y = model.predict(test_x, verbose=1)
        pred_y = K.round(prob_y)

    elif model_name == 'LGBM':
        f = open(f'{model_name}.pkl','rb')
        model = pkl.load(f)

        prob_y = model.predict_proba(test_x)
        pred_y = model.predict(test_x)

    accuracy, precision, recall, f1_score = score(test_y,pred_y,prob_y)

    return accuracy, precision, recall, f1_score
