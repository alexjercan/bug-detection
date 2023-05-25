import string


def solve(test: str) -> str:
    configuration, moves = test.split("\n\n", maxsplit=1)
    moves = moves.splitlines(keepends=False)

    stacks = []
    for line in zip(*configuration.splitlines(keepends=False)):
        stack = [block for block in reversed(line) if block in string.ascii_uppercase]

        if stack:
            stacks.append(stack)

    for move in moves:
        _, count, _, source_stack, _, destination_stack = move.split(" ")
        source_stack = int(source_stack) - 1
        destination_stack = int(destination_stack) - 1
        count = int(count)

        popped = []
        for _ in range(count):
            crate = stacks[source_stack].pop()
            popped.append(crate)

        for crate in reversed(popped):
            stacks[destination_stack].append(crate)

    top_crates = [stack[-1] for stack in stacks]

    return "".join(top_crates)
