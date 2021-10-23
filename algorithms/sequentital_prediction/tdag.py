import os
import sys
parentDir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
newPath=os.path.join(parentDir, 'database')
sys.path.append(newPath)

from predictor_base import preidctor_base
from sequence import sequence

class tdag_node:
    
    def __init__(self, symbol, parent):
        self.symbol = symbol
        self.in_count = 0
        self.out_count = 0
        self.children = {}
        self.parent = parent.copy()
        self.parent.append(symbol)
        self.score = 0

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children

    def get_score(self):
        return self.score

    def set_score(self, score):
        self.score = score

    def get_symbol(self):
        return self.symbol

    def get_in_count(self):
        return self.in_count

    def get_out_count(self):
        return self.out_count



    def add_child(self, symbol):
        child = self.children.get(symbol,None)
        if child == None:
            child = tdag_node(symbol,self.parent)
            self.children[symbol] = child

        self.out_count += 1
        child.in_count += 1

        return child

    def has_child(self, child):
        return self.children.get(child,None) != None


class tdag(preidctor_base):

    def __init__(self):
        super().__init__()
        self.tag = "TDAG"
        self.root = tdag_node(0,[])
        self.size = 1
        self.state = []
        self.map_node = {}
        self.max_tree_height = 6


    def train(self, sequences):
        
        for seq in sequences:
            self.state = []
            self.state.append(self.root)

            for item in seq.get_itemsets():
                new_state = []
                new_state.append(self.root)

                for node in self.state:
                    if len(node.get_parent()) <= self.max_tree_height:

                        if node.has_child(item) == False:
                            self.size += 1


                        child = node.add_child(item)
                        parent = tuple(child.get_parent())
                        self.map_node[parent] = child

                        new_state.append(child)
                
                self.state = new_state

    
        self.state = []


    def predict(self, target):
        
        predicted = sequence()

        symbols = []
        symbols.append(0)

        # symbols.append(item for item in target.get_itemsets())
        for item in target.get_itemsets():
            symbols.append(item)

        key = tuple(symbols)

        context = self.map_node.get(key, None)

        while context == None and len(symbols) > 1:

            del symbols[1]

            key = hash(type(symbols))

            context = self.map_node.get(key, None)

            if context != None and len(context.get_children()) == 0:
                context = None

        
        if context != None:
            
            best_candidate = None
            second_candidate = None

            for it in context.get_children().items():
                score = it[1].get_in_count() / context.get_out_count()
                it[1].set_score(score)

                if best_candidate == None or best_candidate.get_score() < score:
                    second_candidate = best_candidate
                    best_candidate = it[1]

                elif second_candidate == None or second_candidate.get_score() < score:
                    second_candidate = it[1]

                
            threshold = 0

            if best_candidate != None and (second_candidate == None or best_candidate.get_score() - second_candidate.get_score() >= threshold):
                predicted.add_itemset(best_candidate.get_symbol())

            
        return predicted


if __name__ == "__main__":

    seq1 = sequence()
    seq1.itemsets = [1,2,3,4]

    seq2 = sequence()
    seq2.itemsets = [1,2,5,4]

    train = [seq1,seq2]

    p = tdag()
    p.train(train)

    seq3 = sequence()
    seq3.itemsets = [2,3]

    predicted = p.predict(seq3)

    print(predicted.itemsets)



            
