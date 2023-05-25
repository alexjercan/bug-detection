def solve(test: str) -> str:
    chunks = test.strip().split("\n\n")
    current = 0
    maximum = 0

    for chunk in chunks:
        numbers = map(int, chunk.splitlines())
        for number in numbers:
            current += number

        if current > maximum:
            maximum = current

    return str(maximum)
