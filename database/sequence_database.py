from sequence import sequence
import os
import sys

sys.path.insert(1, './..')

class sequence_database:
    
    def __init__(self):
        self.sequences = []
        self.size = 0
        self.num_itemset = 0

    def load_data(self, input):
        with open(input, 'r') as fin:
            lines = fin.read().splitlines()

            for line in lines:
                if len(line):
                    self.add_sequence(line.split(" "))


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
            else:
                itemset.append(int(token))

    def get_sequences(self):
        return self.sequences

    def get_size(self):
        return self.size

    def get_itemset_size(self):
        return self.num_itemset




    