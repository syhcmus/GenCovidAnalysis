import math


from utils.bitmap import bitmap
from utils.prefix import prefix
from pattern.itemset import itemset

class spam():

    def __init__(self):

        self.vertical_db = {}
        self.last_bits_index = []
        self.bitmap_size = 0
        self.minsup = 0
        self.fout = None


    def run(self, input, output, minsup):
        
        self.spam(input, output, minsup)

       


    def spam(self, input, output, minsup):

        self.fout = open(output, "w")
        

        with open(input,'r') as fin:
            lines = fin.read().splitlines()

            bit_index = -1

            for line in lines:
                if len(line):
                    
                    for token in line.split(" "):
                        if token == '-1':
                            bit_index += 1

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


            
            for item in self.vertical_db:
                prefix_item = prefix()
                prefix_item.add_itemset(itemset(item))

                bitmap_item = self.vertical_db[item]

                self.prune(prefix_item,bitmap_item,frequent_items,frequent_items,item,2)

        self.fout.close()


    def prune(self, prefix_item, bitmap, s_items, i_items, considering_item, size):


        # S-step
        s_temp = []
        s_temp_bitmap = []   

        for item in s_items:
            new_bitmap = bitmap.create_s_bitmap(self.vertical_db[item], self.last_bits_index, self.bitmap_size)     

            if(new_bitmap.get_support() >= self.minsup):
                s_temp.append(item)
                s_temp_bitmap.append(new_bitmap)

        for item,new_bitmap in zip(s_temp,s_temp_bitmap):

            new_prefix = prefix_item.clone()
            new_prefix.add_itemset(itemset(item))

            self.save_pattern(new_prefix,new_bitmap)

            self.prune(new_prefix,new_bitmap,s_temp,s_temp,item,size+1)

        # I-step

        i_temp = []
        i_temp_bitmap = []

        for item in i_items:
            if item > considering_item:

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
    output = 'data/output.txt'
    s = spam()
    s.run(input,output,0.5)
    run_time = (time.time() - start_time) * 1000
    
    print(f"--- {run_time} ms ---" )
    

