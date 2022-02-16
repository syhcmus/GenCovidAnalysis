import pandas as pd
import numpy as np

from calculate_occurence import calculate_occurrences
from model import *

def main():

    # model_name = 'CNN'
    model_name = 'LGBM'

    print("Load train dataset")
    filename = "train.csv"
    filepath = '../../input/' + filename
    train_df = pd.read_csv(filepath)
    train_x,train_y = calculate_occurrences(train_df)

    print("Training ...")
    train(train_x,train_y,model_name=model_name)

    print("Load test dataset")
    filename = "test.csv"
    filepath = '../../input/' + filename
    test_df = pd.read_csv(filepath)
    test_x,test_y = calculate_occurrences(test_df)

    print("Predicting ...")
    acc, pre, re, f1 = predict(test_x, test_y,model_name=model_name)

    print('Accuracy: ', acc)
    print('Precision: ', pre)
    print('Recall: ', re)
    print('F1 Score: ', f1)


if __name__ == '__main__':
    main()

    


