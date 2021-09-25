
from utils.left_rule import left_rule

class right_rule(left_rule):

    def __init__(self, itemset, tid, tid_ij, occur):
        super().__init__(itemset, tid, tid_ij)
        self.occur = occur


    def get_occurence(self):
        return self.occur

    