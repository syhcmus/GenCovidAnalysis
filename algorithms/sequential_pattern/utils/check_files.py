
f1 = open("/home/sy/Desktop/project/GenCovidAnalysis/data/MT745584TKS,CM-SPAM,ERMIER,PM.txt", "r")
f2 = open("/home/sy/Desktop/project/GenCovidAnalysis/data/transformed_data.txt", "r")

lines1 = f1.readlines()
lines2 = f2.readlines()

size = min(len(lines1), len(lines2))



print(size)


for index in range(size):
    if lines1[index] != lines2[index]:
        print(index)
        print(lines1[index])
        print(lines2[index])