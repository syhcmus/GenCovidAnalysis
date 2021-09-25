from utils.equivalent_class import equivalent_class

class left_store:

    def __init__(self):
        self.store = {}

    def add_rule(self, rule, itemset,tid,occur):

        index = len(itemset)

        e_map = self.store.get(index, None)
        if e_map == None:
            e_map = {}
            self.store[index] = e_map

        hash_key = hash(tuple(itemset))
        
        l = e_map.get(hash_key, None)
        if l == None:
            eclass = equivalent_class(itemset, tid, occur)
            l = []
            e_map[hash_key] = l
            l.append(eclass)
            eclass.add_rule(rule)
        else:
            for eclass in l:
                if eclass.get_itemset == itemset:
                    eclass.add_rule(rule)
                    return
        
            eclass = equivalent_class(itemset, tid, occur)
            l.append(eclass)
            eclass.add_rule(rule)


    def get_store(self):
        return self.store

