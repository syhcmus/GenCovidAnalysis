import os
import sys
parentDir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
newPath=os.path.join(parentDir, 'database')
sys.path.append(newPath)

from predictor_base import preidctor_base
from sequence import sequence
# from utils.paramable import paramable


class dg_arcs:

    def __init__(self,dest):
        self.dest = dest
        self.support = 1

    def set_dest(self,dest):
        self.dest = dest

    def get_dest(self):
        return self.dest

    def get_support(self):
        return self.support

    def increase_support(self):
        self.support += 1

class dg_node:
    
    def __init__(self, value):
        self.value = value
        self.arcs = []
        self.total_support = 0

    def get_arc_count(self):
        return len(self.arcs)

    def get_arcs(self):
        return self.arcs

    def get_total_support(self):
        return self.total_support


    def update_arc(self,target):
        is_found = False
        for arc in self.arcs:
            if arc.get_dest() == target:
                arc.increase_support()
                is_found = True

        
        if is_found == False:
            self.arcs.append(dg_arcs(target))

    def increase_total_support(self):
        self.total_support += 1

            


class dg(preidctor_base):
    
   
    
    def __init__(self, look_ahead=4):
        super().__init__()
        self.map_node = {}
        self.tag = "DG"
        self.look_ahead = look_ahead


    def train(self, sequences):
        
        window_size = self.look_ahead

        for seq in sequences:

            items = seq.get_itemsets()
            
            for i in range(0, len(items)-1):
                item = items[i]
                node = self.map_node.get(item, None)
                if node == None:
                    node = dg_node(item)
                node.increase_total_support()

                k = i + 1
                while k < i+1+window_size and k < len(items):
                    node.update_arc(items[k])

                    k += 1

                self.map_node[item] = node
    
    def predict(self, target):
        
        threshod = 0.15

        node = None
        offset = 0
        seq_size = target.get_size()
        
        while node == None and offset < seq_size:

            last_item = target.get_itemset(seq_size - (1 + offset))
            node = self.map_node.get(last_item,None)

            offset += 1

        if node == None:
            return sequence()

        max = 0
        best = 0
        
        for arc in node.get_arcs():
            score = arc.get_support() / node.get_total_support()

            if score >= threshod and score > max:
                max = score
                best = arc.get_dest()

        if best == 0:
            return sequence()

        predicted = sequence()
        predicted.add_itemset(best)

        return predicted


if __name__ == '__main__':
    seq = sequence()
    seq.itemsets = [1,2,3,4]

    seq1 = sequence()
    seq1.itemsets = [1,2,5,4]

    train = [seq, seq1]

    seq2 = sequence()
    seq2.itemsets = [2,3]

    test = [seq2]

    predictor = dg()

    predictor.train(train)

    result = predictor.predict(seq2)

    print(result.itemsets)



    