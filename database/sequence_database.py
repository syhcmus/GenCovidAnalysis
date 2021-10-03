from time import perf_counter
from sequence import sequence
import os
import sys
import random

sys.path.insert(1, './..')

class sequence_database:
    
    def __init__(self, max_count=0):
        self.sequences = []
        self.size = 0
        self.num_itemset = 0
        self.max_count = max_count

    def load_data(self, input):
        with open(input, 'r') as fin:
            lines = fin.read().splitlines()

            for line in lines:
                if len(line):
                    if self.max_count == 0:
                        self.add_sequence(line.split(" "))
                    elif self.size < self.max_count:
                        self.add_truncated_sequence(line.split(" "))


    def add_truncated_sequence(self, tokens):
        seq = sequence()
        seq.set_id(self.size)

        self.size += 1

        for token in tokens:
            token = token.strip()
            if len(token) > 0:
                if int(token) > 0:
                    seq.add_itemset(int(token))
        
        self.sequences.append(seq)
        self.num_itemset += 1

    
    def shuffle(self):
        random.shuffle(self.sequences)





    def add_sequence(self, tokens):
        seq = sequence()
        seq.set_id(self.size)

        self.size += 1

        itemset = []

        for token in tokens:
            if token == '-1':
                    self.num_itemset += 1
                    seq.add_itemset(itemset)
                    itemset = []
                
            elif token == '-2':
                self.sequences.append(seq)
                self.num_itemset += 1

            else:
                item = token.strip()
                if len(item) > 0:
                    itemset.append(int(token))

    def get_sequences(self):
        return self.sequences

    def get_size(self):
        return self.size

    def get_itemset_size(self):
        return self.num_itemset

    def clone(self):
        clone_database = sequence_database()
        clone_database.sequences = self.sequences.copy()
        clone_database.size = self.size
        return clone_database

    def set_attr(self, sequences, size):
        self.sequences = sequences
        self.size = size

    def train_test_split(self, ratio):
        relative_ratio = int(self.size * ratio)

        train_set = self.sequences[:relative_ratio].copy()

        test_set = self.sequences[relative_ratio:].copy()
        
        return (train_set,test_set)






    