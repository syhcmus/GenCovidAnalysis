
from matplotlib.pyplot import axis
import pandas as pd

from generate_features import generate_features
from model import predict, train

def main():
    
    print("Load train dataset")
    filename = "train.csv"
    filepath = '../../input/' + filename
    train_df = pd.read_csv(filepath)
    train_x, train_y = train_df.drop('Indicator', axis=1), train_df['Indicator']
    train_x = generate_features(train_x, filename='train_features.csv')

    print("Training")
    train(train_x, train_y)


    print("Load test dataset")
    filename = "test.csv"
    filepath = '../../input/' + filename
    test_df = pd.read_csv(filepath)
    test_x, test_y = test_df.drop('Indicator', axis=1), test_df['Indicator']
    test_x = generate_features(test_x, filename='test_features.csv')

    print("Predicting")
    acc, pre, re, f1 = predict(test_x, test_y)

    print('Accuracy: ', acc)
    print('Precision: ', pre)
    print('Recall: ', re)
    print('F1 Score: ', f1)



if __name__ == '__main__':
    main()
