def solve(test: str) -> str:
    chunks = test.strip().split("\n\n")

    values = []
    for chunk in chunks:
        numbers = map(int, chunk.splitlines())
        current = 0
        for number in numbers:
            current += number

        values.append(current)

    top = sum(reversed(sorted(values)[:3]))
    return str(top)
