import asyncio
import json
import os
import tempfile
from textwrap import indent

import click
import jsonlines
from tqdm.asyncio import tqdm as atqdm

PPX_SAMPLE_JSONL = "preprocessed_samples.jsonl"


def load_dataset(dataset: str) -> list[dict]:
    with open(dataset, "r") as f:
        return json.load(f)["dataset"]


def read_target_directory(target_directory: str) -> list[dict]:
    assert os.path.exists(os.path.join(target_directory, PPX_SAMPLE_JSONL)), (
        f"File {os.path.join(target_directory, PPX_SAMPLE_JSONL)} does not exist"
    )
    with jsonlines.open(os.path.join(target_directory, PPX_SAMPLE_JSONL)) as reader:
        return list(reader)


def get_eval_code(assertion: str, signature: str, io_pairs: str) -> str:
    io_pairs_dict = json.loads(io_pairs)
    signature = signature.split("\n")[0]
    assertion = assertion.replace("return_value", "result").replace("assert", "return")
    eval_code = f"""from typing import *
{signature}
{indent(assertion, " " * 4)}

true_cnt = 0
false_cnt = 0
"""
    for i, o in zip(io_pairs_dict["inputs"], io_pairs_dict["outputs"]):
        input_args_str = ", ".join(repr(arg) for arg in i)
        eval_code += f"v = postcondition({input_args_str}, {repr(o)})\n"
        eval_code += "if v:\n    true_cnt += 1\nelse:\n    false_cnt += 1\n"
    eval_code += "print(f'{true_cnt} {false_cnt}', flush=True)\n"
    return eval_code


async def run_code(code: str) -> tuple[int, int]:
    with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".py") as tmp:
        tmp.write(code)
        tmp.flush()
        tmp_path = tmp.name
        proc = await asyncio.subprocess.create_subprocess_exec(
            "python3",
            tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=30)
        except asyncio.TimeoutError:
            proc.terminate()
            await proc.wait()
            return 0, 0
        if proc.stdout is None:
            return 0, 0
        stdout = await proc.stdout.read()
        stderr = await proc.stderr.read() if proc.stderr else b""
        if stderr:
            print(f"Stderr: {stderr}")

        stdout_str = stdout.decode().strip()
        if not stdout_str:
            return 0, 0

        processed = stdout_str.split()
        if len(processed) != 2:
            print(f"Warning: Unexpected output format: {processed}")
            return 0, 0

        return int(processed[0]), int(processed[1])


async def evaluate_one_assertion(data: dict, evalplus_data: list[dict]) -> dict:
    assertion = data["postcondition_alone"]
    problem_idx = int(data["task_id"].split("/")[-1])
    problem = evalplus_data[problem_idx]
    assert problem["problem_id"] == problem_idx

    # Check completeness
    eval_code = get_eval_code(assertion, problem["signature"], problem["input_output"])
    true_cnt_correct, false_cnt_correct = await run_code(eval_code)

    # Check soundness
    eval_code = get_eval_code(
        assertion, problem["signature"], problem["mutated_input_output"]
    )
    true_cnt_mutated, false_cnt_mutated = await run_code(eval_code)

    total = true_cnt_correct + false_cnt_correct
    if total == 0:
        total = 1  # Prevent division by zero

    result = {
        "is_complete": false_cnt_correct == 0 and true_cnt_correct > 0,
        "is_sound": true_cnt_mutated == 0 and false_cnt_mutated > 0,
        "complete_ratio": true_cnt_correct / total,
        "sound_ratio": false_cnt_mutated / total,
        "true_cnt_correct": true_cnt_correct,
        "false_cnt_correct": false_cnt_correct,
        "true_cnt_mutated": true_cnt_mutated,
        "false_cnt_mutated": false_cnt_mutated,
    }
    return result


def aggregate_results(results: list[dict], exp_name: str) -> dict:
    completeness_ratio = sum(result["is_complete"] for result in results) / len(results)
    soundness_ratio = sum(result["is_sound"] for result in results) / len(results)
    average_complete_ratio = sum(result["complete_ratio"] for result in results) / len(
        results
    )
    average_sound_ratio = sum(result["sound_ratio"] for result in results) / len(
        results
    )
    return {
        "exp_name": exp_name,
        "completeness_ratio": completeness_ratio,
        "soundness_ratio": soundness_ratio,
        "average_complete_ratio": average_complete_ratio,
        "average_sound_ratio": average_sound_ratio,
    }


async def evaluate_target_directory(target_directory: str, dataset: str) -> list[dict]:
    data = read_target_directory(target_directory)
    evalplus_data = load_dataset(dataset)
    tasks = [evaluate_one_assertion(d, evalplus_data) for d in data]
    return await atqdm.gather(*tasks)


@click.command()
@click.argument("target_directory", type=click.Path(exists=True, file_okay=False))
@click.option("--dataset", type=click.Path(exists=True), required=True)
@click.option("--exp_name", type=str, required=True)
def main(target_directory: str, dataset: str, exp_name: str):
    results = asyncio.run(evaluate_target_directory(target_directory, dataset))
    with open(os.path.join(target_directory, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(target_directory, "aggregated_result.json"), "w") as f:
        agg_result = aggregate_results(results, exp_name)
        json.dump(agg_result, f, indent=4)
    print(agg_result)


if __name__ == "__main__":
    main()
