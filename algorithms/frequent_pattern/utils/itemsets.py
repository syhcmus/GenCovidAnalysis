

class itemsets:

    def __init__(self) -> None:
        self.itemsets = [[]]
        self.count = 0


    def add_itemset(self, itemset, k):
        while(len(self.itemsets) <= k):
            self.itemsets.append([])

        self.itemsets[k].append(itemset)
        self.count += 1


    def get_itemsets(self):
        return self.itemsets

    def get_count(self):
        return self.count

    def decrease_count(self):
        self.count -= 1

