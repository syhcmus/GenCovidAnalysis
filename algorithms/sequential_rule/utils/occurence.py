

class occurence:

    def __init__(self, first_itemset, last_itemset) -> None:
        self.first_itemset = first_itemset
        self.last_itemset = last_itemset

    def __str__(self) -> str:
        return self.first_itemset +" "+self.last_itemset

    def set_last_itemset(self, last_itemset):
        self.last_itemset = last_itemset

    def get_first_itemset(self):
        return self.first_itemset

    def get_last_itemset(self):
        return self.last_itemset
