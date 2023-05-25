def solve(test: str) -> str:
    numbers = test.strip().splitlines(keepends=False)
    maximum = 0
    current = 0

    for number in numbers:
        if number == "":
            if current > maximum:
                maximum = current
            current = 0
            continue

        current += int(number)

    if current > maximum:
        maximum = current

    return str(maximum)
