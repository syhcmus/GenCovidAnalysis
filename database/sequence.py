

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

    def set_itemsets(self, itemsets):
        self.itemsets = itemsets

    def get_size(self):
        return len(self.itemsets)

    def get_itemsets(self):
        return self.itemsets

    def get_last_itemsets(self, length, offset):
        truncated_seq = sequence()
        size = self.get_size() - offset

        if(len(self.itemsets) == 0):
            return None

        elif length > size:
            truncated_itemsets = self.itemsets[size-length:size].copy()
            truncated_seq.set_itemsets(truncated_itemsets)
        else:
            truncated_list = self.itemsets[size-length:size].copy()
            truncated_seq.set_itemsets(truncated_list)

        return truncated_seq

        

    

    

    