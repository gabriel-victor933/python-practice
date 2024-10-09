import math
import sys

def plusMinus(arr):
    length = len(arr)
    positive = 0 
    negative = 0 
    zero = 0 
    for number in arr:
        if number < 0:
            negative += 1
        elif number > 0:
            positive += 1
        else: 
            zero += 1

    print("{:.6f}".format(positive/length))
    print("{:.6f}".format(negative/length))
    print("{:.6f}".format(zero/length))
        
def miniMaxSum(arr):
    min = arr[0]
    max = arr[0]
    sum = arr[0]

    for number in arr[1:]:

        sum += number

        if number > max:
            max = number
        elif number < min:
            min = number

    print(f"{sum - max} {sum - min}")

def timeConversion(s):
    # Write your code here
    time = s[0:-2]
    post_meridian = "PM" == s[-2:]
    
    [hour,minutes,seconds] = time.split(':')

    if post_meridian:
        hour = int(hour) + 12
        if hour == 24:
            hour = 12
    else:
        if hour == '12':
            hour = 0

    return "{:02s}:{:s}:{:s}".format(str(hour), minutes, seconds)

def breakingRecords(scores):
    count_least = 0
    count_most = 0

    most = scores[0]
    least = scores[0]

    for number in scores[1:]:
        if number > most:
            most = number
            count_most += 1
        elif number < least:
            least = number
            count_least += 1

    return [count_most, count_least]

def camelCase(text):
    [operation, objectType, words] = text.split(';')

    result = ''

    if operation == 'S':
        if objectType == 'M':
            words = words[0:-2]

        for letter in words:
            if letter.isupper():
                result += " " + letter.lower()
            else:
                result += letter

    elif operation == 'C':
        separate_words = words.split(' ')
        
        for i,word in enumerate(separate_words):
            if i == 0 and objectType != 'C':
                result += word
            else:
                result += word.capitalize()
        
        if objectType == 'M':
            result += "()"

    return result.strip()

## complexidade N²
def divisibleSumPairs(n, k, ar):
    sum = 0

    for i in range(0,n-1): 
        for j in range(i+1,n):
            if (ar[i] + ar[j])%k == 0:
                sum += 1
        
    return sum

def matchingStrings(strings, queries):
    result = [0]*len(queries)
    
    for i,query in enumerate(queries):
        for string in strings:
            if string == query:
                result[i] += 1

    return result

## complexidade N²
def findMedian(arr):
    ordened_arr = mergeSort(arr)
    return ordened_arr[math.floor(len(ordened_arr)/2)]

def mergeSort(arr):
    if len(arr) <= 1:
        return arr
    
    middle = math.floor(len(arr)/2)

    left = mergeSort(arr[0:middle])
    right = mergeSort(arr[middle:])

    i = 0
    j = 0
    result = []

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    to_extend = left[i:] if i < len(left) else right[j:]

    return result + to_extend

def lonelyinteger(a):
    numbers = {}

    for number in a:
        if numbers.get(number) is not None:
            numbers[number] += 1
        else:
            numbers[number] = 1

    for [key,value] in numbers.items():
        if value == 1:
            return key

def gradingStudents(grades):
    for i in range(0,len(grades)):
        if grades[i] < 38:
            continue

        next_multiple = (grades[i]//5 + 1)*5

        if next_multiple - grades[i] < 3:
            grades[i] = next_multiple
        

    return grades

def flippingBits(n):    
    binary = []
    while n > 0:
        rest = n % 2
        n = n // 2
        print(rest,n)
        binary.append(rest)

    binary.reverse()

    listof_binary = [0]*(32 - len(binary)) + binary

    decimal = 0

    for i in range(0,len(listof_binary)):
        if listof_binary[len(listof_binary) -1 - i] == 0:
            decimal += 2**i
        
    return decimal

def diagonalDifference(arr):
    n = len(arr)
    control = 0
    sum_diag_prin = 0
    sum_diag_sec = 0

    while control < n:
        sum_diag_prin += arr[control][control]
        sum_diag_sec += arr[n - control - 1][control]
        control += 1

    return abs(sum_diag_prin - sum_diag_sec)

def countingSort(arr):
    max = arr[0]
    for i in range(1,len(arr)):
        if arr[i] > max:
            max = arr[i]

    arr_integers = [0]*100

    for number in arr:
        arr_integers[number] += 1

    return len(arr_integers)

def countingValleys(steps, path):
    level = 0
    count_valleys = 0
    
    for step in path:
        if step == 'D':
            level -= 1
        elif step == 'U':
            if level == -1: 
                count_valleys += 1
            level += 1

    return count_valleys

def pangrams(s):
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    for letter in s:
        alphabet = alphabet.replace(letter.lower(),"")

    print(alphabet)
    return "pangram" if len(alphabet) == 0 else "not pangram"

def marsExploration(s):

    count = 0
    for i,letter in enumerate(s):
        letter_index = i % 3

        if letter_index == 1:
           if letter != "O":
               count += 1
        else: 
            if letter != "S":
               count += 1
        

    return count

    n = len(matriz)  # Tamanho da matriz (n x n)
    
    for i in range(n // 2):  # Iterar até a metade da matriz
        # Troca o elemento da linha i com o elemento da linha n-i-1 na coluna c
        matriz[i][c], matriz[n - i - 1][c] = matriz[n - i - 1][c], matriz[i][c]

def flippingMatrix(matrix):
    
    n = len(matrix)//2
    sum = 0 
    for i in range(0,n):
        for j in range(0,n):
            sum += max(matrix[i][j], matrix[i][2*n - 1 -j], matrix[2*n - 1 - i][j], matrix[2*n - 1 - i][ 2*n - 1 - j])

    return sum

def twoArrays(k, A, B):
    b_ordened = mergeSort(B)
    a_ordened = mergeSort(A)
    a_ordened.reverse()
    
    for i in range(0,len(A)):
        if a_ordened[i] + b_ordened[i] < k:
            return 'NO'

    return 'YES'

def birthday(s, d, m):
    count = 0
    for i in range(0, len(s)-m+1):
        segment_sum = sum(s[i:i+m])
        if segment_sum == d:
            count += 1

    return count

def sockMerchant(n, ar):
    socks = {}

    for i in range(n):
        if socks.get(ar[i]) is None:
            socks[ar[i]] = 1
        else:
            socks[ar[i]] += 1
    
    count = 0
    for value in socks.values():
        count += value//2

    return count

def migratoryBirds(arr):
    
    birds = {}

    for id in arr:
        if birds.get(id) is None:
            birds[id] = 1
        else:
            birds[id] +=  1

    max = -1
    id = None
    for [item,value] in birds.items():
        if value > max:
            max = value
            id = item
        elif value == max:
            if item < id: 
                id = item
    
    return id

def maximumPerimeterTriangle(sticks):

    sticks = mergeSort(sticks)

    valid_trian = []
    for i in range(len(sticks) - 2):
        print(sticks[i:i+3])
        if sticks[i] + sticks[i+1] > sticks[i+2]:
            valid_trian.append(sticks[i:i+3])

    if len(valid_trian) == 0:
        return [-1]

    valid_trian.reverse()
    max = [0]*3
    print(valid_trian)
    
    for trian in valid_trian: 
        if sum(trian) > sum(max):
            max = trian
        elif sum(trian) == sum(max):
            if trian[2] > max[2]:
                max = trian
            elif trian[0] > max[0]:
                max = trian
        else: 
            return max

    return max

def findZigZagSequence(a, n):
    a.sort()
    
    mid = int(n/2)
    a[mid], a[n-1] = a[n-1], a[mid]
    
    st = mid + 1
    ed = n - 2

    while st <= ed:
        a[st], a[ed] = a[ed], a[st]
        st = st + 1
        ed = ed - 1

    for i in range(n):
        if i == n-1:
            print(a[i])
        else:
            print(a[i])
    return

def pageCount(n, p):
    page_turn_begin = math.floor((p)/2)

    p -= 1 if n % 2 == 0 else 0

    page_turn_end = math.floor((n-p)/2)
    
    return min(page_turn_begin,page_turn_end)

def getTotalX(a, b):

    max_factor = min(b)

    list_factors = []

    for factor in range(1,max_factor+1):
        count = 0
        for number  in a:
            if factor % number == 0:
                count += 1
        
        if count == len(a):
            list_factors.append(factor)

    new_factors = []
    for factor in list_factors:
        count = 0
        for number in b:
            if number % factor == 0:
                count += 1

        if count == len(b):
            new_factors.append(factor)
    
    return len(new_factors)

def pickingNumbers(a):

    count = 1   
    maxs = []
    print(a)
    for i in range(0,len(a)-1):
        print(a[i],a[i+1],abs(a[i] - a[i+1]) <= 1,count)
        if abs(a[i] - a[i+1]) <= 1:
            count += 1
        else:
            maxs.append(count)
            count = 1

    maxs.append(count)

    return max(maxs)

def pickingNumbers2(a):
    # Cria uma lista para armazenar a contagem de cada número no array
    freq = [0] * 101  # O problema define que os números estão entre 0 e 100
    
    # Conta a frequência de cada número
    for num in a:
        freq[num] += 1
    
    max_length = 0
    
    # Verifica a maior soma de frequências entre pares de números consecutivos
    for i in range(1, 101):
        max_length = max(max_length, freq[i] + freq[i-1])
    
    return max_length

def rotateLeft(d, arr):

    n = len(arr)

    new_arr = [0]*n

    for i in range(n):
        if i - d < 0:
            new_arr[n + i - d] = arr[i]
        else:
            new_arr[i - d] = arr[i]

    return new_arr

def kangaroo(x1, v1, x2, v2):
    
    if v2 - v1 == 0:
        return "NO"

    jumps = (x1 - x2)/(v2 - v1)
    print(jumps)
    return "YES" if jumps.is_integer() and jumps > 0 else "NO"

def anagram(s):
    if len(s) % 2 != 0: 
        return -1

    letters = list(s)

    n = len(letters)//2

    min = letters[0:n]

    max = letters[n:]

    count = 0

    for letter in min:
        if letter not in max:
            count += 1
        else:
            max.remove(letter)


    return count

# AUMENTA O LIMITE MÁXIMO DO INTEIRO
sys.set_int_max_str_digits(2147483647)
def fibonacciModified2(t1, t2, n):
    if n == 1: 
        return 0
    elif n == 2:
        return 1

    return fibonacciModified2(t1,t2, n-2) + fibonacciModified2(t1,t2,n-1)**2

def fibonacciModified(t1, t2, n):

    table = [t1, t2]

    for i in range(2,n):
        new_value = table[i-2] + table[i-1]**2
        table.append(new_value)

    return table.pop()

def bigSorting(unsorted):
    if len(unsorted) <= 1:
        return unsorted

    middle = len(unsorted)//2

    left = bigSorting(unsorted[0:middle])
    right = bigSorting(unsorted[middle:])

    i = 0
    j = 0
    sorted = []
    while i < len(left) and j < len(right):

        if (len(left[i]) < len(right[j])) or (len(left[i]) == len(right[j]) and int(left[i]) <= int(right[j])):
            sorted.append(left[i])
            i += 1
        else:
            sorted.append(right[j])
            j += 1


    sorted.extend(left[i:])
    sorted.extend(right[j:])
    
    return sorted

def separateNumbers(s):

    def partitionArr(numbers,get):
        sequence = []
        for i in range(0,len(numbers),get):
            
            sequence.append("".join(numbers[i:i+get]))
        return sequence

    def isSequencial(arr):
        for i in range(len(arr) -1):
            
            if int(arr[i]) + 1 != int(arr[i+1]):
                ## se arr[i] for composto de nove a sequencia tem que ser rearranja para um ordem acima
                if len(arr[i].replace("9","")) == 0 and (len(arr[i]) == len(arr[i+1])):
                    
                    new_arr = list("".join(arr[i+1:]))

                    sequence = [arr[i]] + partitionArr(new_arr,len(arr[i+1]) + 1)

                    return isSequencial(sequence)

                return False

            if arr[i].startswith('0') and len(arr[i]) > 1:
                return False

        return True

    if len(s) <=1:
        print("NO")
        return

    max_length_split = len(s)//2

    numbers = list(s)
    
    for take in range(1,max_length_split+1):
        sequence = partitionArr(numbers,take)
        
        if isSequencial(sequence):
            print("YES " + sequence[0])
            return
        
            
    print("NO")


separateNumbers('1234')
separateNumbers('91011')
separateNumbers('99100')
separateNumbers('101103')
separateNumbers('010203')
separateNumbers('13')
separateNumbers('1')
separateNumbers('99910001001')
