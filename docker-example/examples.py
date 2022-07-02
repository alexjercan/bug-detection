# Example 1
A = map(input().split())
print(A[0])

# Example 2
A = "abcde"
for i in range(5):
    A[i] = "x"
print(A)

# Example 3
A = input()
B = int(input())
print(A + B)

# Example 4
A = input().split(",")
p = 1
for a in A:
    p = p * a
print(p)

# Example 5
A = input().split(",")
p = 1
for a in A:
    p = p * int(a)
print(p)

