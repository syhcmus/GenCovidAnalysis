
f1 = open("output.txt", "r")
f2 = open("output1.txt", "r")

lines1 = f1.readlines()
lines2 = f2.readlines()

size = min(len(lines1), len(lines2))

print(size)
print(len(lines1))
print(len(lines2))


# for index in range(size):
#     if lines1[index] != lines2[index]:
#         print(index)
#         print(lines1[index])
#         print(lines2[index])