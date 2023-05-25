def solve(test: str) -> str:
    points = {"A": 1, "B": 2, "C": 3}
    beats = {"A": "C", "B": "A", "C": "B"}
    loses = {"A": "B", "B": "C", "C": "A"}

    lines = test.strip().splitlines(keepends=False)

    score = 0
    for line in lines:
        fst, out = line.split(" ")

        if out == "X":
            score += points[beats[fst]]
        elif out == "Y":
            score += 3 + points[fst]
        elif out == "Z":
            score += 6 + points[loses[fst]]

    return str(score)
