from file_splitter import file_spliter
from novel_fast_vector import create_vects_and_country_wise_distance_matrix
from representative_sequence_finder import find_representative_sequence
from model import train, transform_data, predict

def main():
  

    print("Processing data")
    file_spliter()
    create_vects_and_country_wise_distance_matrix()
    find_representative_sequence()


    single_position = False # predict single position or not
    start = 23825 # start site of interests 
    end = 23829 # end site of interests
    # (start, end) is in the set (8445,8449), (19610,19614), (24065,24069), (23825,23829)
    pos = 3 # single position of genome sequence to predict, position is in the set (1,2,3,4,5)
    # position is 1 to predict 5th, position is 2 to predict the 4th and so on
    model_name = 'CNN'
    

    print("Transform data")
    transform_data(start,end,pos,single_position)

    print("Training")
    train(model_name,single_position)

    print("Predicting")
    predict(model_name,single_position)

    eval = predict(model_name,single_position)
    test_accuracy = eval[1]
    print(f'Accuracy: {round(100 * test_accuracy, 2)}%')




if __name__ == "__main__":
    main()