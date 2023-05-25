def solve(test: str) -> str:
    lines = test.strip().splitlines(keepends=False)

    result = 0
    for line in lines:
        int1, int2 = line.split(",", maxsplit=1)
        low1, high1 = map(int, int1.split("-", maxsplit=1))
        low2, high2 = map(int, int2.split("-", maxsplit=1))

        if max(low1, low2) <= min(high1, high2):
            result += 1

    return str(result)
