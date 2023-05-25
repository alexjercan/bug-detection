def solve(test: str) -> str:
    points = {"X": 1, "Y": 2, "Z": 3}
    beats = {"X": "C", "Y": "A", "Z": "B"}
    mapping = {"X": "A", "Y": "B", "Z": "C"}

    lines = test.strip().splitlines(keepends=False)

    score = 0
    for line in lines:
        fst, snd = line.split(" ")

        if fst == beats[snd]:
            score += 6
        elif fst == mapping[snd]:
            score += 3

        score += points[snd]

    return str(score)
