from typing import Dict, List


def color_source(source_code: str, i1: int, i2: int, color: str = "red"):
    lines = source_code.splitlines(keepends=True)

    text = ""
    for i, line_str in enumerate(lines):
        for char in line_str:
            norm_color = "black"
            if char == " ":
                char = "•"
                norm_color = "lightgrey"
            if char == "\n":
                char = "↵\n"
                norm_color = "lightgrey"
            col = color if i1 <= i <= i2 else norm_color
            text += f'<span style="color:{col};">{char}</span>'

    return "<pre>" + text + "</pre>"


def generate_html_output(
    examples: Dict[str, List], test_results: Dict[int, List]
) -> Dict[str, List]:
    # Display the source code. First show the buggy submission with red lines for the bug
    # Then show the passing submission with green lines. Then show the predictions
    # with red lines if the change did not work, and with green lines if the change
    # made the tests pass
    results = []
    passed = []
    for i, fail, pass_, i1, i2, j1, j2, predictions in zip(
        examples["index"],
        examples["fail"],
        examples["pass"],
        examples["i1"],
        examples["i2"],
        examples["j1"],
        examples["j2"],
        examples["predicted"],
    ):
        fail_html = color_source(fail, i1, i2, color="red")
        pass_html = color_source(pass_, j1, j2, color="green")

        html = ""
        html += f"<h2>Example {i}</h2>"

        html += "<h6>Original Source Code</h6>"
        html += fail_html

        html += "<h6>Changed Source Code</h6>"
        html += pass_html

        any_correct = False
        for j, (pred, (_, test)) in enumerate(zip(predictions, test_results[i])):
            color = "green" if test["passed"] else "red"
            # TODO: how can I make this better
            pred_html = color_source(pred, 0, len(pred.splitlines()), color=color)

            html += f"<h6>Predicted Source Code {j}</h6>"
            html += pred_html

            any_correct = any_correct or test["passed"]

        results.append(html)
        passed.append(any_correct)

    return {"html": results, "any_correct": passed}


def compute_bug_type_pass(example: Dict[str, List]) -> Dict[str, List]:
    results = []
    for j1, j2, pass_, language in zip(
        example["j1"], example["j2"], example["pass"], example["language"]
    ):
        line = "\n".join(pass_.splitlines()[j1:j2])

        if language == "Python":
            results.append(
                "input"
                if "input" in line
                else "output"
                if "print" in line
                else "algorithm"
            )
        elif language == "C++":
            results.append(
                "input"
                if ("cin" in line or "scanf" in line)
                else "output"
                if ("cout" in line or "printf" in line)
                else "algorithm"
            )
        else:
            raise NotImplementedError(f"{language} not implemented yet")

    return {"pass_bug_type": results}


def compute_bug_type_predicted(example: Dict[str, List]) -> Dict[str, List]:
    results = []
    for i1, i2, fail, predicted, language in zip(
        example["i1"],
        example["i2"],
        example["fail"],
        example["predicted"],
        example["language"],
    ):
        diff_len = len(predicted[0].splitlines()) - len(fail.splitlines())
        line = "\n".join(predicted[0].splitlines()[i1 : i2 + diff_len])

        if language == "Python":
            results.append(
                "input"
                if "input" in line
                else "output"
                if "print" in line
                else "algorithm"
            )
        elif language == "C++":
            results.append(
                "input"
                if ("cin" in line or "scanf" in line)
                else "output"
                if ("cout" in line or "printf" in line)
                else "algorithm"
            )
        else:
            raise NotImplementedError(f"{language} not implemented yet")

    return {"predicted_bug_type": results}


def compute_bug_type(example: Dict[str, List], which: str) -> Dict[str, List]:
    if which == "pass":
        return compute_bug_type_pass(example)

    if which == "predicted":
        return compute_bug_type_predicted(example)

    raise NotImplementedError(f"{which} not implemented, use pass or predicted")
