from sklearn import pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import pandas as pd
import pickle as pkl



def score(test_y, pred_y, prob_y):
    '''
    Score the prediction
    Input: true label, predicted label, probility of predicted label
    Output: score of metrics
    '''
    accuracy = accuracy_score(test_y, pred_y)
    precision = precision_score(test_y, pred_y)
    recall = recall_score(test_y, pred_y)
    f1 = f1_score(test_y, pred_y)
    
    return accuracy, precision, recall, f1


def train(train_x, train_y):
    '''
    Training model
    Input: train features, train lable
    Output: Pipeline fitted train dataset
    '''
    extraTree = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=1)
    
    clf = LGBMClassifier(objective='binary')

    grid_params = {
    'learning_rate': [ 0.01, 0.05],
    'num_leaves': [50,100],
    'max_depth' : [5,10,15],
    'min_data_in_leaf':[5, 10]
    }

    model = RandomizedSearchCV(clf,grid_params,verbose=1,cv=10,n_iter=10)

    steps = [('SFM', SelectFromModel(estimator=extraTree)),
             ('Scaler', StandardScaler()),
             ('CLF', clf)]

    pipeline = Pipeline(steps)

    pipeline.fit(train_x, train_y)


    f = open(f'LGBM.pkl','wb')
    pkl.dump(pipeline, f)

    # return pipeline


def predict(test_x, test_y):
    '''
    Predict and evaluate the prediction
    Input: pipeline, test features, test label
    Output: Score of metrics evaluating the prediction
    '''
   
    f = open(f'LGBM.pkl','rb')
    pipeline = pkl.load(f)

    pred_y = pipeline.predict(test_x)
    prob_y = pipeline.predict_proba(test_x)
    acc, pre, re, f1 = score(test_y, pred_y, prob_y)

    return acc, pre, re, f1



    # prob_y = prob_y[:,1]
    # draw_roc_curve(test_y, prob_y, dir)
    # prob_y = pd.DataFrame(prob_y, columns=['Probability of 1'])
    # test_y = pd.DataFrame(test_y.values, columns=['Indicator'])
    # result = pd.concat([test_y, pred_y, prob_y], axis=1, sort=False)
    # result.to_csv(dir + 'Prediction.csv', index=False)

    # print('Accuracy: ', acc)
    # print('Precision: ', pre)
    # print('Recall: ', re)
    # print('F1 Score: ', f1)
    # print('ROC AUC Score', roc)   