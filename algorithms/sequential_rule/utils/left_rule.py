
class left_rule:

    def __init__(self, itemset, tid, tid_ij) -> None:
        self.itemset = itemset
        self.tid = tid
        self.tid_ij = tid_ij

    def get_itemset(self):
        return self.itemset

    def get_tid(self):
        return self.tid

    def get_tid_ij(self):
        return self.tid_ij

    def __str__(self) -> str:
        if len(self.itemset) == 1:
            return str(self.itemset[0])  + "=>" + "..."
        return ','.join(self.itemset) + "=>" + "..."