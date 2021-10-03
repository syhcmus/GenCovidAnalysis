import math
import sys
from typing import Sequence

import os
parentDir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
newPath=os.path.join(parentDir, 'database')
sys.path.append(newPath)

from sequence_database import *

import time


from utils.bitmap import bitmap
from utils.prefix import prefix
from pattern.itemset import itemset



class cmspam():

    def __init__(self):

        self.vertical_db = {}
        self.last_bits_index = []
        self.bitmap_size = 0
        self.minsup = 0
        self.fout = None
        self.cmap_s = {}
        self.cmap_i = {}
        self.database = sequence_database()



    def run(self, input, output, minsup):
        
        self.cmspam(input, output, minsup)

       


    def cmspam(self, input, output, minsup):

        self.fout = open(output, "w")
        

        self.database.load_data(input)
        self.bitmap_size = self.database.get_itemset_size()
        previous_seq_last_index = 0
        for sid in range(len(self.database.get_sequences())):
            seq = self.database.get_sequences()[sid]
            self.last_bits_index.append(seq.get_size()-1 + previous_seq_last_index)
            previous_seq_last_index = self.last_bits_index[-1] + 1
            

            for tid in range(seq.get_size()):
                for item in seq.get_itemset(tid):
                    bitmap_item = self.vertical_db.get(item, None)
                    if bitmap_item == None:
                        bitmap_item = bitmap(self.bitmap_size)
                        self.vertical_db[item] = bitmap_item

                    bitmap_item.register_bit(sid, tid, self.last_bits_index)

        self.minsup = math.ceil(minsup * self.database.get_size())


        
    
        self.vertical_db = dict(sorted(self.vertical_db.items()))

        frequent_items = []
        
        for item in list(self.vertical_db.keys()):
            bitmap_item = self.vertical_db[item]

            if(bitmap_item.get_support() >= self.minsup):
                frequent_items.append(item)
                self.save_pattern(item, bitmap_item)
            else:
                del self.vertical_db[item]

        # cmap


        for seq in self.database.get_sequences():
            already_processed = set()
            i_processed = {}            

            for tid in range(seq.get_size()):
                for i in range(len(seq.get_itemset(tid))):
                    item = seq.get_itemset(tid)[i]

                    i_set = i_processed.get(item,None)
                    if i_set == None:
                        i_set = set()
                        i_processed[item] = i_set

                    bitmap_item = self.vertical_db.get(item, None)
                    if bitmap_item == None or bitmap_item.get_support() < self.minsup:
                        continue

                    already_processed_item = set()

                    for j in range(i+1, len(seq.get_itemset(tid))):
                        item_j = seq.get_itemset(tid)[j]

                        bitmap_item_j = self.vertical_db.get(item_j, None)
                        if bitmap_item_j == None or bitmap_item_j.get_support() < minsup:
                            continue                        

                        if item_j not in i_set:
                            cmap = self.cmap_i.get(item_j,None)
                            if cmap == None:
                                cmap = {}
                                self.cmap_i[item] = cmap

                            support = cmap.get(item_j,None)
                            if support == None:
                                cmap[item_j] = 1
                            else:
                                support += 1
                                cmap[item_j] = support

                            i_set.add(item_j)


                    if tid < seq.get_size()- 1:
                        for tid1 in range(tid+1, seq.get_size()):
                            for item_j in seq.get_itemset(tid1):

                                if item in already_processed:
                                    break

                                cmap = self.cmap_s.get(item,None)
                                if cmap == None:
                                    cmap = {}
                                    self.cmap_s[item] = cmap

                                support = cmap.get(item_j,None)
                                if support == None:
                                    cmap[item_j] = 1
                                else:
                                    support += 1
                                    cmap[item_j] = support

                                already_processed_item.add(item_j)


                    already_processed.add(item)
        
        for item in self.vertical_db:
            prefix_item = prefix()
            prefix_item.add_itemset(itemset(item))

            bitmap_item = self.vertical_db[item]

            self.prune(prefix_item,bitmap_item,frequent_items,frequent_items,item,2,item)

        self.fout.close()


    def prune(self, prefix_item, bitmap, s_items, i_items, considering_item, size, last_appended_item):

        self.count += 1
        print(self.count)

        # S-step
        s_temp = {}
        # s_temp_bitmap = []   

        map_support_s = self.cmap_s.get(last_appended_item,None)

        for item in s_items:

            # cmap prunning
            if map_support_s == None:
                continue
            
            support = map_support_s.get(item)
            if support == None or support < self.minsup:
                continue

            new_bitmap = bitmap.create_s_bitmap(self.vertical_db[item], self.last_bits_index, self.bitmap_size)


            if(new_bitmap.get_support() >= self.minsup):
                s_temp[item] = new_bitmap



        for item,new_bitmap in s_temp.items():
            new_prefix = prefix_item.clone()
            new_prefix.add_itemset(itemset(item))

            self.save_pattern(new_prefix,new_bitmap)

        
            self.prune(new_prefix,new_bitmap,s_temp.keys(),s_temp.keys(),item,size+1, item)


        # I-step

        i_temp = {}
        # i_temp_bitmap = []

        map_support_i = self.cmap_i.get(last_appended_item,None)


        for item in i_items:
            if item > considering_item:

                # cmap prunning
                if map_support_i == None:
                    continue
                
                support = map_support_i.get(item)
                if support == None or support < self.minsup:
                    continue


                

                new_bitmap = bitmap.create_i_bitmap(self.vertical_db[item], self.last_bits_index, self.bitmap_size)

                if new_bitmap.get_support() >= self.minsup:
                    i_temp[item] = new_bitmap

        for item,new_bitmap in i_temp.items():
            new_prefix = prefix_item.clone()
            new_prefix.get_itemset(len(prefix_item) - 1).add_item(item)

            self.save_pattern(new_prefix,new_bitmap)

            self.prune(new_prefix,new_bitmap,s_temp.keys(),i_temp.keys(),item,size+1)


    def save_pattern(self,pattern, bitmap):

        result = ""
        
        if isinstance(pattern, int):
            result = f"{pattern} -1 #SUP: {bitmap.get_support()}\n"
        elif isinstance(pattern,prefix):
            for itemset in pattern.get_itemsets():
                for item in itemset.get_items():
                    result += f"{item} "
                
                result += "-1 "

            result += f"#SUP: {bitmap.get_support()}\n"


        self.fout.write(result)


import time

if __name__ == '__main__':
    
    start_time = time.time()

    newPath=os.path.join(parentDir, 'data') 
    input = os.path.join(newPath, 'input.txt') # data.input.txt
    output = os.path.join(newPath, 'output1.txt') # data/output1.txt

    s = cmspam()
    s.run("/home/sy/Desktop/project/GenCovidAnalysis/data/transformed_data.txt",output,0.98)
    run_time = (time.time() - start_time) * 1000
    
    print(f"--- {run_time} ms ---" )
    

