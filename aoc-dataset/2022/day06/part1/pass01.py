def solve(test: str) -> str:
    for i in range(len(test) - 3):
        if len(set(test[i : i + 4])) == 4:
            break

    return str(i + 4)
