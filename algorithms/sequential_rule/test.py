

class myclass:
    def __init__(self, num) -> None:
        self.num = num
    
    def __eq__(self, o: object) -> bool:
        return self.num - o.num

    def set_num(self, n):
        self.num = n

    


# e1 = myclass(3)
# e2 = myclass(2)

# l = [e1, e2]

# for e in sorted(l, key=lambda e:e.num, reverse=True):
#     print(e.num)


a = myclass(1)
a.set_num(2)
print(a.num)