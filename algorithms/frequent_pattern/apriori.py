import math
import os
import sys

parentDir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
newPath=os.path.join(parentDir, 'database')
sys.path.append(newPath)

from sequence_database import sequence_database
from utils.itemset import itemset

class apriori:

    def __init__(self) -> None:
        self.fout =  None
        self.database = sequence_database(max_count=500)
        self.minsup = 0

    def run(self, input, output, minsup):
        self.fout = open(output,'w')

        self.database.load_data(input)

        self.minsup = math.ceil(minsup * self.database.get_size())

        map_item_count = self.database.get_map_item_count()

        k = 1

        map_item_count = dict(sorted(map_item_count.items()))

        large_one_itemset = []
        for entry in map_item_count.items():
            item = entry[0]
            support = entry[1]
            
            if support >= self.minsup:
                large_one_itemset.append(item)
                self.save_itemset(item, support)

        map_item_count = None

        
        

        large_itemset = []
        k = 2

        while True:
            Ck = [] 

            if k == 2:
                Ck = self.generate_CK_size_2(large_one_itemset)
            else:
                Ck = self.generate_Ck_size_k(large_itemset)

            for seq in self.database.get_sequences():
                if seq.get_size() < k:
                    continue

                for candidate in Ck:
                    pos = 0

                    for item in seq.get_itemsets():

                        if item == candidate.get(pos):
                            pos += 1

                            if pos == len(candidate.get_itemset()):
                                candidate.increase_support()
                                break

                        elif item > candidate.get(pos):
                            break
                


            large_itemset = []
            for candidate in Ck:
                if candidate.get_support() >= self.minsup:
                    large_itemset.append(candidate)
                    self.save_itemset(candidate,candidate.get_support())


            k += 1

            if len(large_itemset) == 0:
                break

        self.fout.close()


    def generate_CK_size_2(self, large_itemset):
        Ck = []

        for i in range(len(large_itemset)):
            item_i = large_itemset[i]
            for j in range(i+1, len(large_itemset)):
                item_j = large_itemset[j]

                new_itemset = itemset()
                new_itemset.set_itemset([item_i, item_j])

                Ck.append(new_itemset)

        return Ck


    def generate_Ck_size_k(self, large_itemset):

        Ck = []
        already_processed = set()

        for i in range(len(large_itemset)):
            loop_i = False
            itemset_i = large_itemset[i].get_itemset()
            
            for j in range(i+1, len(large_itemset)):
                loop_j = False
                itemset_j = large_itemset[j].get_itemset()

                for k in range(len(itemset_i)):
                    if k == len(itemset_i) - 1:
                        if itemset_i[k] >= itemset_j[k]:
                            loop_i = True
                            break

                    elif itemset_i[k] < itemset_j[k]:
                        loop_j = True
                        break

                    elif itemset_i[k] > itemset_j[k]:
                        loop_i = True
                        break

                if loop_i == True:
                    break
                elif loop_j == True:
                    continue



                new_itemset = itemset_i.copy()
                new_itemset.append(itemset_j[-1])

                hash_key = hash(tuple(new_itemset))

                if self.is_contain_all_frequent_subset(new_itemset, large_itemset) and hash_key not in already_processed:

                    already_processed.add(hash_key)

                    new_itemset_oj = itemset()
                    new_itemset_oj.set_itemset(new_itemset)

                    Ck.append(new_itemset_oj)

        return Ck

                

    def is_contain_all_frequent_subset(self, candidate, large_itemset):
        
        for pos in range(len(candidate)):
            first = 0
            last = len(large_itemset) - 1

            is_found = False

            while first <= last:

                middle = (first + last) // 2

                comparison = self.is_same(large_itemset[middle].get_itemset(), candidate, pos)
                if comparison < 0:
                    first = middle + 1
                elif comparison > 0:
                    last = middle - 1
                else:
                    return True

            if is_found == False:
                return False

        
            


    
    def is_same(self, itemset1, itemset2, pos):
        j = 0
        for i in range(len(itemset1)):
            if j == pos:
                j += 1

            if itemset1[i] == itemset2[j]:
                j += 1

            elif itemset1[i] > itemset2[j]:
                return 1

            else:
                return -1

        
        return 0
        

    def save_itemset(self, item, support):
        self.fout.write(f"{item} #SUP: {support}\n")
        


if __name__ == '__main__':

    newPath = os.path.join(parentDir, 'data') 
    input = os.path.join(newPath, 'transformed_data.txt') # data.input.txt
    output = os.path.join(newPath, 'output.txt') # data/output.txt

    apr = apriori()
    apr.run(input, output, 0.1)
    
    