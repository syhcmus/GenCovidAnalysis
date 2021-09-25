
class equivalent_class:
    def __init__(self, itemset, tid, occur):
        self.itemset = itemset
        self.tid = tid
        self.occurnce = occur
        self.rules = []


    def get_rules(self):
        return self.rules

    def set_rules(self, rules):
        self.rules = rules

    def add_rule(self, rule):
        self.rules.append(rule)

    def get_itemset(self):
        return self.itemset

    def get_tid(self):
        return self.tid

    def get_occurence(self):
        return self.occurnce
        