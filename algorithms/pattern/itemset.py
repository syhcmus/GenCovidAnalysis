
class itemset:

    def __init__(self,item):
        self.items = []
        self.items.append(item)


    def add_item(self, item):
        self.items.append(item)

    def get_items(self):
        return self.items

    def get(self, index):
        return self.items[index]

    def get_size(self):
        return len(self.items)