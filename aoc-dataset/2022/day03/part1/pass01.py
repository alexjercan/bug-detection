import string


def solve(test: str) -> str:
    priorities = {
        c: v
        for v, c in enumerate(string.ascii_lowercase + string.ascii_uppercase, start=1)
    }

    lines = test.strip().splitlines(keepends=False)

    priority = 0
    for line in lines:
        midpoint = len(line) // 2
        part1, part2 = set(line[:midpoint]), set(line[midpoint:])

        common = "".join(part1.intersection(part2))

        priority += priorities.get(common, 0)

    return str(priority)
