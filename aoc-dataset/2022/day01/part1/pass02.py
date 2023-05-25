def solve(test: str) -> str:
    chunks = test.strip().split("\n\n")
    maximum = 0

    for chunk in chunks:
        current = 0
        numbers = map(int, chunk.splitlines())
        for number in numbers:
            current += number

        if current > maximum:
            maximum = current

    return str(maximum)
