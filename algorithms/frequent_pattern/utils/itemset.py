
class itemset:

    def __init__(self) -> None:
        self.itemset = []
        self.support = 0


    def __str__(self) -> str:
        if len(self.itemset) == 1:
            return self.itemset[0]

        

        return ' '.join([str(e) for e in self.itemset])


    def get_itemset(self):
        return self.itemset

    def set_itemset(self, itemset):
        self.itemset = itemset.copy()

    def set_support(self, support):
        self.support = support

    def get_support(self):
        return self.support

    def get(self, index):
        return self.itemset[index]

    def increase_support(self):
        self.support += 1

    
    def clone_exclude_item(self, excluded_item):
        new_itemset = []

        for item in self.itemset:
            if item != excluded_item:
                new_itemset.append(item)

        new_itemset_oj = itemset()
        new_itemset_oj.set_itemset(new_itemset)

        return new_itemset_oj

    def clone_exclude_itemset(self, excluded_itemset):
        new_itemset = []

        for item in self.itemset:
            if item not in excluded_itemset:
                new_itemset.append(item)

        new_itemset_oj = itemset()
        new_itemset_oj.set_itemset(new_itemset)

        return new_itemset_oj
        
