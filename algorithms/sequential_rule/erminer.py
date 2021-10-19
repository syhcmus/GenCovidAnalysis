
# importing sys
import sys
import math
import time

import os
parentDir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
newPath=os.path.join(parentDir, 'database')
sys.path.append(newPath)
  


from sequence_database import sequence_database
from utils.occurence import occurence
from utils.sparse_matrix import sparse_matrix
from utils.equivalent_class import equivalent_class
from utils.left_rule import left_rule
from utils.right_rule import right_rule
from utils.left_store import left_store

class erminer:

    def __init__(self):
        self.database = sequence_database()
        self.minsup = 0
        self.minconf = 0
        self.map_item = {}
        self.matrix = sparse_matrix()
        self.left_store = left_store()
        self.fout = None
        
        

    def run(self, input, output, min_conf, min_sup):
        
        self.erminer(input, output, min_conf, min_sup)

    def erminer(self, input, output, min_conf, min_sup):
        self.database.load_data(input)
        self.fout = open(output, "w")

        self.minsup = math.ceil(min_sup * self.database.get_size())
        self.minconf = min_conf

        self.cal_frequence_item(self.database)
        self.generate_sparse_matrix(self.database)
        

        left_eclass = {}
        right_eclass = {}

        self.matrix.set_matrix(dict(sorted(self.matrix.get_matrix().items())))            

        for items_i in self.matrix.get_matrix().items():

            item_i = items_i[0]
            occurence_i = self.map_item[item_i]

            for items_j in items_i[1].items():

                if items_j[1] < self.minsup:
                    continue

                item_j = items_j[0]
                occurence_j = self.map_item[item_j]

                tid_ij = set()
                tid_ji = set()

                if len(occurence_i) < len(occurence_j):
                    self.cal_tid(occurence_i, occurence_j, tid_ij, tid_ji)
                else:
                    self.cal_tid(occurence_j, occurence_i, tid_ji, tid_ij)


                count_ij = len(tid_ij)
                if count_ij >= self.minsup:
                    conf_ij = count_ij / len(occurence_i)

                    itemset_i = [item_i]
                    itemset_j = [item_j]


                    if conf_ij >= min_conf:
                        self.save_rule(tid_ij, conf_ij, itemset_i, itemset_j)

                    self.register_rule(item_i, occurence_i, tid_ij, item_j, occurence_j, left_eclass, right_eclass )

                
                count_ji = len(tid_ji)

                if count_ji >= self.minsup:

                    conf_ji = count_ji / len(occurence_j)

                    itemset_i = [item_i]
                    itemset_j = [item_j]

                    if conf_ji >= min_conf:
                        self.save_rule(tid_ji, conf_ji, itemset_j, itemset_i)

                    self.register_rule(item_j, occurence_j, tid_ji, item_i, occurence_i, left_eclass, right_eclass )                
            
        

        
        for l_eclass in left_eclass.values():
            if len(l_eclass.get_rules()) > 1:

                l_eclass.set_rules(sorted(l_eclass.get_rules(), key=lambda eclass: eclass.get_itemset()[0]))
                
                self.expand_left(l_eclass)


        right_eclass = dict(sorted(right_eclass.items()))

        for r_eclass in right_eclass.values():

            if len(r_eclass.get_rules()) > 1:
                r_eclass.set_rules(sorted(r_eclass.get_rules(), key= lambda eclass: eclass.get_itemset()[0]))

            self.expand_right(r_eclass)


        for e_map in self.left_store.get_store().values():
            for e_list in e_map.values():
                for e_class in e_list:
                    if len(e_class.get_rules()) > 1:
                        e_class.set_rules(sorted(e_class.get_rules(), key= lambda e: e.get_itemset()[-1]))
                        self.expand_left(e_class)

        self.fout.close()




    def cal_frequence_item(self, database):
        for sid in range(database.get_size()):
            seq = database.get_sequences()[sid]

            for tid in range(seq.get_size()):
                itemset = seq.get_itemset(tid)

                for item in itemset:

                    occurs = self.map_item.get(item, None)

                    if occurs == None:
                        occurs = {}
                        occurs[sid] =  occurence(tid, tid)
                        self.map_item[item] = occurs
                    else:
                        occur = occurs.get(sid, None)
                        if occur == None:
                            occurs[sid] = occurence(tid, tid)
                        else:
                            occur.set_last_itemset(tid)
                            


                    

    def generate_sparse_matrix(self, database):

        for seq in database.get_sequences():

            already_processed_item = set()

            for itemset in seq.get_itemsets():
                for item in itemset:
                    if item in already_processed_item or len(self.map_item.get(item)) < self.minsup:
                        continue

                    co_occuring_item = set()

                    for co_itemset in seq.get_itemsets():
                        for co_item in co_itemset:
                            if co_item == item or co_item in co_occuring_item or len(self.map_item.get(co_item)) < self.minsup:
                                continue

                            self.matrix.increase_count(item, co_item)
                            co_occuring_item.add(co_item)
                        
                    
                    already_processed_item.add(item)


    def cal_tid(self, occurence_i, occurence_j, tid_ij, tid_ji):

        for occur_i in occurence_i.items():
            tid = occur_i[0]

            oc_j = occurence_j.get(tid, None)
            if oc_j != None:
                oc_i = occur_i[1]

                if oc_j.get_first_itemset() < oc_i.get_last_itemset():
                    tid_ji.add(tid)
                if oc_i.get_first_itemset() < oc_j.get_last_itemset():
                    tid_ij.add(tid)



    def save_rule(self, tid_ij, conf_ij, itemset_i, itemset_j):
        
        ante = []
        for itemset in itemset_i:
            ante.append(itemset)
        
        conseq = []
        for itemset in itemset_j:
            conseq.append(itemset)

        sup = len(tid_ij)
        conf = conf_ij

        if len(ante) == 1:
            ante = str(ante[0])
        else:
            ante = ','.join([str(e) for e in ante])
            

        if len(conseq) == 1:
            conseq = str(conseq[0])
        else:
            conseq = ','.join([str(e) for e in conseq])


        rule = f"{ante} ==> {conseq} #SUP: {sup} #CONF: {conf}\n"
        self.fout.write(rule)


    def register_rule(self, item_i, occurence_i, tid_ij, item_j, occurence_j, l_eclass, r_eclass):
        tid_i = occurence_i.keys()
        tid_j = occurence_j.keys()

        left_class = l_eclass.get(item_j, None)
        if left_class == None:
            left_class = equivalent_class([item_j], tid_j, occurence_j )
            l_eclass[item_j] = left_class

        l_rule = left_rule([item_i], tid_i, tid_ij)
        left_class.add_rule(l_rule)

        

        right_class = r_eclass.get(item_i, None)
        if right_class == None:
            right_class = equivalent_class([item_i], tid_i, occurence_i)
            r_eclass[item_i] = right_class
        
        r_rule = right_rule([item_j], tid_j, tid_ij, occurence_j)
        right_class.add_rule(r_rule)


    def expand_left(self, left_eclass):
        for i in range(len(left_eclass.get_rules())-1):
            rule_i = left_eclass.get_rules()[i]
            d = rule_i.get_itemset()[-1]

            rules = equivalent_class(left_eclass.get_itemset(), 
                left_eclass.get_tid(), left_eclass.get_occurence())
            
            for j in range(i+1, len(left_eclass.get_rules())):
                rule_j = left_eclass.get_rules()[j]

                c = rule_j.get_itemset()[-1]

                if self.matrix.get_count(c, d) < self.minsup:
                    continue

                tid_ic = set()

                map_c = self.map_item[c]

                rule_size = len(rule_i.get_tid())

                if rule_size < len(map_c):
                    for tid in rule_i.get_tid():
                        if map_c.get(tid, None) != None:
                            tid_ic.add(tid)

                        rule_size -= 1

                        if len(tid_ic) + rule_size < self.minsup:
                            break

                else:
                    rule_size = len(map_c)
                    for tid in map_c.keys():
                        if tid in rule_i.get_tid():
                            tid_ic.add(tid)

                        rule_size -= 1
                        if len(tid_ic) + rule_size < self.minsup:
                            break
                

                tid_ic_j = set()

                if len(rule_i.get_tid_ij()) < len(map_c):
                    for tid in rule_i.get_tid_ij():
                        occurence_c = map_c.get(tid, None)
                        if occurence_c != None:
                            occurence_j = left_eclass.get_occurence()[tid]
                            if occurence_c.get_first_itemset() < occurence_j.get_last_itemset():
                                tid_ic_j.add(tid)

                else:
                    for item in map_c.items():
                        tid = item[0]
                        if tid in rule_i.get_tid_ij():
                            occurence_c = item[1]
                            occurence_j = left_eclass.get_occurence()[tid]
                            if occurence_c.get_first_itemset() < occurence_j.get_last_itemset():
                                tid_ic_j.add(tid)

                count_ic_j = len(tid_ic_j)
                if count_ic_j >= self.minsup:
                    conf_ic_j = count_ic_j / len(tid_ic)

                    itemset_ic = rule_i.get_itemset().copy()
                    itemset_ic.append(c)

                    new_rule = left_rule(itemset_ic, tid_ic, tid_ic_j)

                    if conf_ic_j >= self.minconf:
                        self.save_rule(tid_ic_j, conf_ic_j, itemset_ic, left_eclass.get_itemset())

                    rules.add_rule(new_rule)

            if len(rules.get_rules()):
                self.expand_left(rules)



    def expand_right(self, right_eclass):
        

        for i in range(len(right_eclass.get_rules())-1):
            rule_i = right_eclass.get_rules()[i]
            d = rule_i.get_itemset()[-1]
            rules = equivalent_class(right_eclass.get_itemset(), right_eclass.get_tid(), right_eclass.get_occurence())

            for j in range(i+1,len(right_eclass.get_rules())):
                rule_j = right_eclass.get_rules()[j]
                c = rule_j.get_itemset()[-1]

                if self.matrix.get_count(c,d) < self.minsup:
                    continue

                tid_i_jc = set()
                map_c = self.map_item[c]

                rule_size = len(rule_i.get_tid_ij())
                if rule_size < len(map_c):
                    for tid in rule_i.get_tid_ij():
                        occurence_c = map_c.get(tid, None)
                        if occurence_c != None:
                            occurence_i = right_eclass.get_occurence()[tid]
                            if occurence_c.get_last_itemset() > occurence_i.get_first_itemset():
                                tid_i_jc.add(tid)

                        rule_size -= 1
                        if len(tid_i_jc)+rule_size < self.minsup:
                            break
                else:
                    rule_size = len(map_c)
                    for item in map_c.items():
                        tid = item[0]
                        
                        if tid in rule_i.get_tid_ij():
                            occurence_c = item[1]
                            occurence_i = right_eclass.get_occurence()[tid]
                            if occurence_c.get_last_itemset() > occurence_i.get_first_itemset():
                                tid_i_jc.add(tid)

                    rule_size -= 1
                    if len(tid_i_jc) + rule_size < self.minsup:
                        break

                if len(tid_i_jc) >= self.minsup:
                    tid_jc = set()
                    occurence_jc = {}  

                    if len(rule_i.get_tid()) < len(map_c):
                        for tid in rule_i.get_tid():
                            occurence_c = map_c.get(tid,None)
                            if occurence_c != None:
                                tid_jc.add(tid)
                                occurence_j = rule_i.get_occurence()[tid]
                                if occurence_c.get_last_itemset() < occurence_j.get_last_itemset():
                                    occurence_jc[tid] = occurence_c
                                else:
                                    occurence_jc[tid] = occurence_j
                    else:
                        for item in map_c.items():
                            tid = item[0]
                            if tid in rule_i.get_tid():
                                tid_jc.add(tid)

                                occurence_c = item[1]
                                occurence_j = rule_i.get_occurence()[tid]
                                if occurence_c.get_last_itemset() < occurence_j.get_last_itemset():
                                    occurence_jc[tid] = occurence_c
                                else:
                                    occurence_jc[tid] = occurence_j

                    
                    conf_i_jc = len(tid_i_jc) / len(right_eclass.get_tid())
                    itemset_jc = rule_i.get_itemset().copy()
                    itemset_jc.append(c)

                    if conf_i_jc >= self.minconf:
                        self.save_rule(tid_i_jc,conf_i_jc,right_eclass.get_itemset(), itemset_jc)

                    r_rule = right_rule(itemset_jc, tid_jc, tid_i_jc, occurence_jc)

                    rules.add_rule(r_rule)

                    l_rule = left_rule(right_eclass.get_itemset(), right_eclass.get_tid(), tid_i_jc)
                    self.left_store.add_rule(l_rule,itemset_jc,tid_jc, occurence_jc)

            if len(rules.get_rules()):
                self.expand_right(rules)

if __name__ == '__main__':


    start_time = time.time()

    newPath=os.path.join(parentDir, 'data') 
    input = os.path.join(newPath, 'transformed_data.txt') 
    output = os.path.join(newPath, 'output.txt') # data/output.txt

    erm = erminer()
    erm.run(input,output,0.5,0.5)
    run_time = (time.time() - start_time) * 1000
    
    print(f"--- {run_time} ms ---" )

