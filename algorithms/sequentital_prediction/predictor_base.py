import abc

class preidctor_base:
    
    def __init__(self):
        self.tag = None

    @abc.abstractmethod
    def train(self, sequences):
        pass

    @abc.abstractclassmethod
    def predict(self, target):
        pass

    def get_tag(self):
        return self.tag
    
    def set_tag(self, tag):
        self.tag = tag

    @abc.abstractclassmethod
    def get_size(self):
        pass

    