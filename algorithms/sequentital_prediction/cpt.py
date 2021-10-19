
import os
import sys
parentDir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
newPath=os.path.join(parentDir, 'database')
sys.path.append(newPath)

from bitarray import bitarray

from predictor_base import preidctor_base
from sequence import sequence
# from utils.paramable import paramable

class prediction_tree:

    def __init__(self) -> None:
        self.support = 0
        self.item = None
        self.children = []
        self.parent = None

    def set_item(self, item):
        self.item = item

    def get_item(self):
        return self.item

    def get_parent(self):
        return self.parent

    def add_child(self, child):
        prediction_tree_oj = prediction_tree()
        prediction_tree_oj.set_item(child)
        new_child = prediction_tree_oj
        new_child.parent = self
        self.children.append(new_child)

    def has_child(self, target):

        for child in self.children:
            if child.item == target:
                return True

        return False


    def get_child(self, target):

        for child in self.children:
            if child.item == target:
                return child
        
        return None
    

    

class cpt(preidctor_base):
    def __init__(self, min_recursion=1,max_recursion=99):
        super().__init__()
        self.tag = "CPT"
        self.root = prediction_tree()
        self.lookup_table = {}
        self.invert_index = {}
        self.count_table = {}
        self.min_recursion = min_recursion
        self.max_recursion = max_recursion
        



    def get_matching_sequences(self, target):

        itemset = target.get_itemsets()
        intersection = None
        
        for item in itemset:
            bit_array = self.invert_index.get(item,None)
            if bit_array != None:
                if intersection == None:
                    intersection = bit_array.copy()
                else:
                    intersection = intersection & bit_array


        if intersection == None or intersection.count(1) == 0:
            return None

        indexes = intersection.search(1)

        return indexes

        


    def update_count_table(self, target, weight, count_table, visited_sid):
        indexes = self.get_matching_sequences(target)

        hash_target = set(target.get_itemsets())

        for index in indexes:
            
            if index in visited_sid:
                continue

            current_node = self.lookup_table[index]

            branch = []
            while current_node.get_parent() != self.root:
                branch.append(current_node.get_item())
                current_node = current_node.get_parent()

            branch.reverse()

            already_processed = hash_target.copy()

            i=0

            while i < len(branch) and len(already_processed) > 0:

                if branch[i] in already_processed:
                    already_processed.remove(branch[i])

                i += 1

            while i < len(branch):

                old_value = 0
                
                if branch[i] in count_table:
                    old_value = count_table[branch[i]]

                count_table[branch[i]] = old_value + weight / len(indexes)

                visited_sid.add(index)
                i += 1
        
    def get_best_sequence(self, count_table):
        max_value = -1
        second_value = -1
        max_item = -1

        for it in count_table.items():

            score = it[1] # confidence

            if score > max_value:
                second_value = max_value
                max_item = it[0]
                max_value = score

            elif score > second_value:
                second_value = score

        predicted = sequence()

        diff = 1 - (second_value / max_value)

        if diff >= 0 or second_value == -1:
            predicted.add_itemset(max_item)

        elif max_item != -1:
            best_score = 0
            best_item = -1

            for it in count_table.items():
                key = it[0]
                value = it[1]

                if max_value == value:
                    if key in self.invert_index:
                        
                        score = value / self.invert_index[key].count(1) # lift
                        
                        if score > best_score:
                            best_score = score
                            best_item = key

            predicted.add_itemset(best_item)

        return predicted


    def train(self, sequences):

        sequence_id = 0
        sequences_size = len(sequences)

        for seq in sequences:
            current_node = self.root

            for item in seq.get_itemsets():

                if self.invert_index.get(item, None) == None:
                    bit_array = bitarray(sequences_size)
                    bit_array.setall(0)
                    self.invert_index[item] = bit_array

                self.invert_index[item][sequence_id] = 1

                if current_node.has_child(item) == False:
                    current_node.add_child(item)

                current_node = current_node.get_child(item)

            self.lookup_table[sequence_id] = current_node
            sequence_id += 1


    
                
    def predict(self, target):

        for item in target.get_itemsets():
            if self.invert_index.get(item, None) == None:
                target.get_itemsets().remove(item)


        predicted = sequence()

        min_recursion = self.min_recursion
        max_recursion = target.get_size()

        if max_recursion > self.max_recursion:
            max_recursion = self.max_recursion

        i = min_recursion
        while i < target.get_size() and predicted.get_size() == 0 and i < max_recursion:

            
            min_size = target.get_size() - i

            sub_sequences = []
            self.recursive_divide(sub_sequences, target, min_size)

            count_table = {}
            visited_sid = set()

            for seq in sub_sequences:
                weight = seq.get_size() / target.get_size()
                self.update_count_table(seq, weight, count_table, visited_sid)

            predicted = self.get_best_sequence(count_table)

            i += 1

        return predicted


    def recursive_divide(self, sequences, target, min_size):
        size = target.get_size()
        sequences.append(target)

        if size <= min_size:
            return

        for i in range(size):
            new_sequence = sequence()
            
            for j in range(size):
                if i != j:
                    new_sequence.add_itemset(target.get_itemset(j))

            self.recursive_divide(sequences,new_sequence,min_size)


if __name__ == '__main__':
    seq = sequence()
    seq.itemsets = [1,2,3,4]

    seq1 = sequence()
    seq1.itemsets = [1,2,3,4]

    seq2 = sequence()
    seq2.itemsets = [1,2,3,4]

    seq3 = sequence()
    seq3.itemsets = [0,1,2,4]

    train = [seq, seq1,seq2,seq3]

    seq_test = sequence()
    seq_test.itemsets = [0,1,2]



    predictor = cpt()

    predictor.train(train)

    result = predictor.predict(seq_test)

    print(result.itemsets)




            



            



    
                    


    


    