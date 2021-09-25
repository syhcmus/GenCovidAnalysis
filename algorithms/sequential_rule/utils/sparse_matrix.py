
class sparse_matrix:

    def __init__(self):
        self.matrix = {}

    def increase_count(self, item, co_item):
        if item >= co_item:
            cmap = self.matrix.get(item)

            if cmap == None:
                cmap = {}
                cmap[co_item] = 1
                self.matrix[item] = cmap
            else:
                count = cmap.get(co_item, 0)
                if count == 0:
                    cmap[co_item] = 1
                else:
                    cmap[co_item] = count + 1

    def get_count(self, item, co_item):
        cmap = self.matrix.get(item, None)

        if cmap == None:
            return 0

        count = cmap.get(co_item, 0)

        return count


    def get_matrix(self):
        return self.matrix

    def set_matrix(self, matrix):
        self.matrix = matrix