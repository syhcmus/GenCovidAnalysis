

class result:

    def __init__(self):
        self.data = {}

    def get(self,stat):
        if self.data.get(stat,None) == None:
            self.data[stat] = 0.0

        return self.data[stat]


    def set(self, stat, value):
        self.data[stat] = value