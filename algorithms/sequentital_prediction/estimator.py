

import os
import sys
from dg import dg
from akom import akom

parentDir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
newPath=os.path.join(parentDir, 'database')
sys.path.append(newPath)

from sequence_database import sequence_database
from utils.profile import profile
from utils.statistic_logger import statistic_logger

class estimator:

    def __init__(self, input, max_count=500):
        self.predictors = []
        self.database = sequence_database(max_count)
        self.stats = None
        self.input = input
        


    def add_pridictor(self, predictor):
        self.predictors.append(predictor)

    def run(self, sample_type = 'kfold', param = 10):

        sample_type = sample_type.lower()

        predictor_tags = []
        for predictor in self.predictors:
            predictor_tags.append(predictor.get_tag())

        self.database.load_data(self.input)

        self.database.shuffle()

        stats_cols = ["success", "failure", "overall"]

        self.stats = statistic_logger(stats_cols, predictor_tags )

        for idx in range(len(self.predictors)):

            if sample_type == 'holdout':
                self.holdout(param, idx)

            elif sample_type == 'kfold':
                self.kfold(param, idx)

            else:
                raise ValueError("sample type is not defined")


        self.final_stat()

        print(self.stats)


    def train(self, train_set, classifier_id):
        self.predictors[classifier_id].train(train_set)
    

    def predict(self, test_set, classifier_id):
        for target in test_set:
            conseq_size = profile.get_int_param("consequent_size")
            window_size = profile.get_int_param("window_size")   

            predicttor_name = self.predictors[classifier_id].get_tag()

            if target.get_size() > conseq_size:

                consequence = target.get_last_itemsets(conseq_size,0)
                final_target = target.get_last_itemsets(window_size, conseq_size)

                predict = self.predictors[classifier_id].predict(final_target)
                

                if self.is_success(consequence,predict):
                    self.stats.increase("success", predicttor_name)

                else:
                    self.stats.increase("failure", predicttor_name)




    def holdout(self, ratio, classifier_id):
        (training_set, test_set) = self.database.train_test_split(ratio)

        self.train(training_set, classifier_id)
        self.predict(test_set, classifier_id)
        
        
        

    def kfold(self, k, classifier_id):
        if k > 1:

            dataset = self.database.clone().get_sequences()

            relative_ratio = 1 / k
            absolute_ratio = int(len(dataset) * relative_ratio)

            for i in range(k):
                pos_start = i * absolute_ratio
                pos_end = pos_start + absolute_ratio

                if i == k-1:
                    pos_end = len(dataset)

                
                train_set = []
                test_set = []

                for j in range(len(dataset)):
                    seq = dataset[j]

                    if j >= pos_start and j < pos_end:
                        test_set.append(seq)
                    else:
                        train_set.append(seq)

                
                self.train(train_set, classifier_id)
                self.predict(test_set, classifier_id)


    def is_success(self, target, predicted):

        for pre in predicted.get_itemsets():
            is_error = True
            for tar in target.get_itemsets():
                if tar == pre:
                    is_error = False


            if is_error == True:
                return False

        return True


    def final_stat(self):
        for predictor in self.predictors:
            
            predictor_tag = predictor.get_tag()

            success = int(self.stats.get("success", predictor_tag))
            failure = int(self.stats.get("failure", predictor_tag))
            no_match = int(self.stats.get("no match", predictor_tag))

            maching_size = success + failure
            testing_size = maching_size + no_match


            self.stats.cal_percent("success", predictor_tag, maching_size)
            self.stats.cal_percent("failure", predictor_tag, maching_size)
            self.stats.cal_percent("no match", predictor_tag, maching_size)

            self.stats.set("overall", predictor_tag, success)
            self.stats.cal_percent("overall", predictor_tag, testing_size)



    
    

if __name__ == "__main__":

    newPath=os.path.join(parentDir, 'data') 
    input = os.path.join(newPath, 'transformed_data.txt') # data/input.txt
    
    est = estimator(input)
    est.add_pridictor(dg())
    est.add_pridictor(akom())

    est.run(sample_type="kfold", param=10)
    # est.run(sample_type="holdout", param=0.8)

