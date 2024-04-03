def fibonacci(n):
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    return sequence

n = int(input("Enter the number of terms: "))
fib_sequence = fibonacci(n)
print(fib_sequence)
