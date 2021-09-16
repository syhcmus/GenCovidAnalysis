from bitarray import bitarray
from bisect import bisect


class bitmap():

    def __init__(self, num_bit) -> None:
        self.bit_arr = bitarray(num_bit)
        self.bit_arr.setall(0)
        self.support = 0
        self.last_sid = -1
        


    def register_bit(self, sid, tid, last_bits_index):

        if sid == 0:
            index = tid
        else:
            index = last_bits_index[sid - 1] + 1 + tid

        self.bit_arr[index] = True

        if self.last_sid != sid:
            self.support += 1
            self.last_sid = sid

    def set_bit(self, index):
        self.bit_arr[index] = True

    def get_bit(self, index):
        return self.bit_arr[index]

    def get_sequence_index(self, bit_index, last_bits_index):

        try:
            return last_bits_index.index(bit_index)
        except ValueError:
            return bisect(last_bits_index, bit_index)

        # left = 0
        # right = len(last_bits_index) - 1
        

        # while right >= left:
            
        #     mid = (left+right)//2

        #     if last_bits_index[mid] == bit_index:
        #         return mid
        #     elif last_bits_index[mid] > bit_index:
        #         right = mid -1
        #     else:
        #         left = mid + 1
            
        # return left



    def get_support(self):
        return self.support

    def set_support(self, support):
        self.support = support


    def increase_support(self):
        self.support += 1

    def get_sequences_index(self, last_bits_index):

        indexes = set()

        for bit_index in self.bit_arr.search(1):
            indexes.add(self.get_sequence_index(bit_index, last_bits_index))

        return indexes

    def get_last_bit_index_sequence(self,sid, last_bits_index):
        return last_bits_index[sid]
        

    def create_s_bitmap(self, bitmap_item, last_bits_index, bitmap_size):

        new_bitmap = bitmap(bitmap_size)
        
        start_index = 0
        index = self.bit_arr.find(1,start_index)
     

        while index >= 0:

            sid = self.get_sequence_index(index, last_bits_index)
            last_bit_index_sequence = self.get_last_bit_index_sequence(sid,last_bits_index)
        
            is_find_match_sequence = False
            index_item = bitmap_item.bit_arr.find(1, index + 1)

            while index_item >= 0 and index_item <= last_bit_index_sequence:

                new_bitmap.set_bit(index_item)
                is_find_match_sequence = True

                index_item = bitmap_item.bit_arr.find(1, index_item + 1)

            if(is_find_match_sequence):
                new_bitmap.increase_support()


            start_index = last_bit_index_sequence + 1
            index = self.bit_arr.find(1,start_index)
            

        return new_bitmap



        

    def create_i_bitmap(self, bitmap_item, last_bits_index, bitmap_size):

        new_bitmap = bitmap(bitmap_size)
        last_sid = -1

        for index in self.bit_arr.search(1):
            if bitmap_item.get_bit(index):
                new_bitmap.set_bit(index)

                sid = self.get_sequence_index(index, last_bits_index)

                if sid != last_sid:
                    new_bitmap.increase_support()
                    last_sid = sid

        return new_bitmap

if __name__ == '__main__':

    a = [True,False,False,False,False,True,False,False,True,False,False,False]
    b = [True,True,True,False,True,True,False,False,True,True,False,False]


    last_bits_index = [3,7,11]

    bitmap_size = 12

    bit_a = bitmap(bitmap_size)
    bit_b = bitmap(bitmap_size)

    for i in range(bitmap_size):
        if a[i]:
            bit_a.set_bit(i)
        if b[i]:
            bit_b.set_bit(i)

    print(bit_a.bit_arr[:])
    print(bit_b.bit_arr[:])


    
    new_s_bitmap = bit_a.create_s_bitmap(bit_b,last_bits_index,bitmap_size)

    new_i_bitmap = bit_a.create_i_bitmap(bit_b,last_bits_index,bitmap_size)

    print(new_s_bitmap.bit_arr[:])











    
