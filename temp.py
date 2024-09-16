a = list(range(9))
b = []
for i in range(2):
    b.append(a.pop(0))
print(a)
print(b)