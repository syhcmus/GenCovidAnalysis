
class prefix:
    
    def __init__(self):
        self.itemsets = []

    def add_itemset(self, itemset):
        self.itemsets.append(itemset)

    def get_itemsets(self):
        return self.itemsets

    def get_itemset(self, index):
        return self.itemsets[index]

    def get_size(self):
        return len(self.itemsets)
    
    def clone(self):
        new_prefix = prefix()

        for itemset in self.get_itemsets():
            new_prefix.add_itemset(itemset)

        return new_prefix