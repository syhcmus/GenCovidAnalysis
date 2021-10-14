
import os
import sys
parentDir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
newPath=os.path.join(parentDir, 'database')
sys.path.append(newPath)

from predictor_base import preidctor_base
from utils.paramable import paramable
from sequence import sequence

class markov_state:
    def __init__(self) -> None:
        self.count = 0
        self.transitions = {}


    def get_count(self):
        return self.count

    def add_transition(self, trainsition):
        support = self.transitions.get(trainsition,None)
        if support == None:
            support = 0
            self.count += 1

        self.transitions[trainsition] = support + 1

    def get_next_state(self):
        best_support = 0
        best_state = None

        for item in self.transitions.items():
            if item[1] > best_support:
                best_support = item[1]
                best_state = item[0]


        return best_state


        

    


class akom(preidctor_base):
    def __init__(self,k=5):
        super().__init__()
        self.k = k
        self.tag = "AKOM"
        self.parameters = paramable()
        self.map_node = {}


    def train(self, sequences):
        for seq in sequences:
            items = seq.get_itemsets()

            for i in range(len(items)-1):
                k = self.k
                if len(items) - i <= k:
                    k = len(items) - i - 1

                for c in range(1, k+1):
                    key = ""
                    for j in range(c):
                        key += f"{items[i+j]} "
                    
                    key = key[0:-1]

                    state = self.map_node.get(key,None)
                    if state == None:
                        state = markov_state()

                    state.add_transition(items[i+c])

                    self.map_node[key] = state


    def predict(self, target):
        k = self.k
        if target.get_size() < k :
            k = target.get_size()

        i = k
        while i > 0:

            key = ""
            for j  in range(target.get_size() - i, target.get_size()):
                key += f"{target.get_itemset(j)} "

            key = key[:-1]

            state = self.map_node.get(key,None)
            if state != None:
                next_state = state.get_next_state()
                predicted  = sequence()
                predicted.add_itemset(next_state)

                return predicted


            i -= 1


        return sequence()



if __name__ == '__main__':
    predictor = akom()

    seq1 = sequence()
    seq1.itemsets = [1,2,3,4]

    seq2 = sequence()
    seq2.itemsets = [1,2,3,4]

    seq3 = sequence()
    seq3.itemsets = [1,2,3,4]

    seq4 = sequence()
    seq4.itemsets = [0,1,2,4]

    train = [seq1,seq2,seq3,seq4]

    predictor.train(train)

    seq_test = sequence()
    seq_test.itemsets = [0,1,2]

    result = predictor.predict(seq_test)
    print(result.get_itemsets())





    

    