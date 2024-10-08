# TODO: Implementar o algoritmo merge sort de cabeça sem nenhuma consulta
import math

def mergeSort(arr):
    if len(arr) <= 1: 
        return arr

    middle = math.floor(len(arr)/2)

    left = mergeSort(arr[0:middle])
    right = mergeSort(arr[middle:])

    i = 0
    j = 0
    result = []

    while(i < len(left) and j < len(right)):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else: 
            result.append(right[j])
            j += 1

    to_extended = left[i:] if i < len(left) else right[j:]

    return result + to_extended

print(mergeSort([3,1,5,12,3,8]))