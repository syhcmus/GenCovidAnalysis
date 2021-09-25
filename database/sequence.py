

class sequence:

    def __init__(self):
        self.itemsets = []
        self.id = None

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def add_itemset(self, itemset):
        self.itemsets.append(itemset)

    def get_itemset(self, index):
        return self.itemsets[index]

    def get_size(self):
        return len(self.itemsets)

    def get_itemsets(self):
        return self.itemsets

    

    

    