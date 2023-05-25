def solve(test: str) -> str:
    for i in range(len(test) - 14):
        if len(set(test[i : i + 14])) == 14:
            break

    return str(i + 14)
