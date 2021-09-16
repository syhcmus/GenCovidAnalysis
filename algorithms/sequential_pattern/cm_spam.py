import math
from sys import version
from typing import Sequence


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



    def run(self, input, output, minsup):
        
        self.cmspam(input, output, minsup)

       


    def cmspam(self, input, output, minsup):

        self.fout = open(output, "w")
        

        with open(input,'r') as fin:
            lines = fin.read().splitlines()

            bit_index = -1

            horizontal_db = []

            for line in lines:
                if len(line):
                
                    transaction = []
                    
                    for token in line.split(" "):
                        
                        transaction.append(int(token))

                        if token == '-1':
                            bit_index += 1

                    horizontal_db.append(transaction)

                    self.last_bits_index.append(bit_index)

            self.bitmap_size = bit_index + 1

            self.minsup = math.ceil(minsup * len(self.last_bits_index))

            sid = 0
            tid = 0

            for line in lines:
                if len(line):

                    for token in line.split(" "):
                        if token == "-1":
                            tid += 1
                        elif token == "-2":
                            sid += 1
                            tid = 0
                        else:
                            item = int(token)

                            bitmap_item = self.vertical_db.get(item, None)
                            if bitmap_item == None:
                                bitmap_item = bitmap(self.bitmap_size)
                                self.vertical_db[item] = bitmap_item

                            bitmap_item.register_bit(sid,tid,self.last_bits_index)
            
        
            self.vertical_db = dict(sorted(self.vertical_db.items()))

            frequent_items = []
            
            for item in list(self.vertical_db.keys()):
                bitmap_item = self.vertical_db[item]

                if(bitmap_item.get_support() >= self.minsup):
                    frequent_items.append(item)
                    self.save_pattern(item, bitmap_item)
                else:
                    del self.vertical_db[item]


            for transaction in horizontal_db:
                already_processed = set()
                i_processed = {}

                for i in range(len(transaction)):
                    item = transaction[i]
                    i_set = i_processed.get(item,None)
                    if i_set == None:
                        i_set = set()
                        i_processed[item] = i_set

                    if item < 0:
                        continue

                    bitmap_item = self.vertical_db.get(item, None)
                    if bitmap_item == None or bitmap_item.get_support() < self.minsup:
                        continue

                    already_processed_B = set()

                    same_item_set = True
                    for j in range(i+1,len(transaction)):
                        item_j = transaction[j]

                        if item_j < 0:
                            same_item_set = False
                            continue

                        bitmap_item_j = self.vertical_db.get(item_j, None)
                        if bitmap_item_j == None or bitmap_item_j.get_support() < minsup:
                            continue

                        cmap = {}
                        if same_item_set:
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

                        elif item_j not in already_processed_B:

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

                            already_processed_B.add(item_j)
                    
                    already_processed.add(item)
            
            for item in self.vertical_db:
                prefix_item = prefix()
                prefix_item.add_itemset(itemset(item))

                bitmap_item = self.vertical_db[item]

                self.prune(prefix_item,bitmap_item,frequent_items,frequent_items,item,2,item)

        self.fout.close()


    def prune(self, prefix_item, bitmap, s_items, i_items, considering_item, size, last_appended_item):


        # S-step
        s_temp = []
        s_temp_bitmap = []   

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
                s_temp.append(item)
                s_temp_bitmap.append(new_bitmap)

        for item,new_bitmap in zip(s_temp,s_temp_bitmap):
            new_prefix = prefix_item.clone()
            new_prefix.add_itemset(itemset(item))

            if new_bitmap.get_support() >= self.minsup:

                self.save_pattern(new_prefix,new_bitmap)


                self.prune(new_prefix,new_bitmap,s_temp,s_temp,item,size+1, item)

        # I-step

        i_temp = []
        i_temp_bitmap = []

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
                    i_temp.append(item)
                    i_temp_bitmap(new_bitmap)

        for item,new_bitmap in zip(i_temp,i_temp_bitmap):
            new_prefix = prefix_item.clone()
            new_prefix.get_itemset(len(prefix_item) - 1).add_item(item)

            self.save_pattern(new_prefix,new_bitmap)

            self.prune(new_prefix,new_bitmap,s_temp,i_temp,item,size+1)


    def save_pattern(self,pattern, bitmap):

        result = ""
        
        if isinstance(pattern, int):
            result = f"{pattern} -1 #SUP: {bitmap.get_support()}"
        elif isinstance(pattern,prefix):
            for itemset in pattern.get_itemsets():
                for item in itemset.get_items():
                    result += f"{item} "
                
                result += "-1 "

            result += f"#SUP: {bitmap.get_support()}"

        

        sequences_index = " ".join([str(x) for x in bitmap.get_sequences_index(self.last_bits_index)])

        result += f" #SID: {sequences_index}\n"

        self.fout.write(result)


import time

if __name__ == '__main__':
    
    start_time = time.time()

    input = 'data/input.txt'
    output = 'data/output1.txt'
    s = cmspam()
    s.run(input,output,0.5)
    run_time = (time.time() - start_time) * 1000
    
    print(f"--- {run_time} ms ---" )
    

