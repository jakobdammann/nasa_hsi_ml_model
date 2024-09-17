import numpy as np

arr = np.array([])
print(arr)

for i in range(10):
    array = [1,2,3,51,12,5]
    arr = np.append(arr, array)
    arr.shape = (-1, len(array))
    print(arr)

print(arr)