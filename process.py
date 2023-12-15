f = open("output.txt", "r")

c_0 = 0
c_1 = 0

for line in f:
    if int(line[0])//1 == 1:
        print(line)
        c_1 += 1
    else:
        c_0 += 1

print("c_0", c_0)
print("c_1", c_1)