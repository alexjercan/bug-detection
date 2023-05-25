import string


def solve(test: str) -> str:
    priorities = {
        c: v
        for v, c in enumerate(string.ascii_lowercase + string.ascii_uppercase, start=1)
    }

    lines = test.strip().splitlines(keepends=False)

    priority = 0
    for i in range(0, len(lines), 3):
        common = "".join(set(lines[i]) & set(lines[1]) & set(lines[2]))

        priority += priorities.get(common, 0)

    return str(priority)
