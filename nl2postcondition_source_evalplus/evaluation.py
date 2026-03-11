import asyncio
import json
import os
import tempfile
from textwrap import indent
from typing import List, Optional

import click
from models import (
    AggregatedResult,
    EvaluationResult,
)
from tqdm.asyncio import tqdm as atqdm

from dataset_paths import get_evalplus_dataset_file

PPX_SAMPLE_JSONL = "preprocessed_samples.jsonl"


def load_dataset(dataset: str) -> dict:
    with open(dataset, "r") as f:
        d = json.load(f)
    return {str(d["problem_id"]): d for d in d}


def read_target_directory(target_directory: str) -> list[dict]:
    assert os.path.exists(os.path.join(target_directory, PPX_SAMPLE_JSONL)), (
        f"File {os.path.join(target_directory, PPX_SAMPLE_JSONL)} does not exist"
    )
    # Lazy import to avoid hard dependency at module import time
    import jsonlines  # type: ignore

    with jsonlines.open(os.path.join(target_directory, PPX_SAMPLE_JSONL)) as reader:
        return list(reader)


def get_eval_code_with_parser(assertion: str, signature: str, args, parser: str) -> str:
    signature = signature.split("\n")[0]
    assertion = assertion.replace("return_value", "result").replace("assert", "return")
    eval_code = f"""from typing import *
import math
import re
{parser}
{signature}
{indent(assertion, " " * 4)}

true_cnt = 0
false_cnt = 0
error_cnt = 0
"""
    io_pairs = json.loads(args)
    for i, o in zip(io_pairs["inputs"], io_pairs["outputs"]):
        eval_code += "try:\n"
        eval_code += f"    v = postcondition(*parser({repr(i)}, {repr(o)}))\n"
        eval_code += "    if v == True:\n        true_cnt += 1\n    else:\n        false_cnt += 1\n"
        eval_code += "except Exception as e:\n    error_cnt += 1\n"
    eval_code += "print(f'{true_cnt} {false_cnt} {error_cnt}', flush=True)\n"
    return eval_code


def get_eval_code_with_io_pairs(assertion: str, signature: str, io_pairs: str) -> str:
    io_pairs_dict = json.loads(io_pairs)
    signature = signature.split("\n")[0]
    assertion = assertion.replace("return_value", "result").replace("assert", "return")
    eval_code = f"""from typing import *
import math
import re
{signature}
{indent(assertion, " " * 4)}

true_cnt = 0
false_cnt = 0
error_cnt = 0
"""
    for i, o in zip(io_pairs_dict["inputs"], io_pairs_dict["outputs"]):
        input_args_str = ", ".join(repr(arg) for arg in i)
        eval_code += "try:\n"
        eval_code += f"    v = postcondition({input_args_str}, {repr(o)})\n"
        eval_code += "    if v == True:\n        true_cnt += 1\n    else:\n        false_cnt += 1\n"
        eval_code += "except Exception as e:\n    raise e\n"
    eval_code += "print(f'{true_cnt} {false_cnt} {error_cnt}', flush=True)\n"
    return eval_code


def get_eval_code(assertion: str, signature: str, args, parser: Optional[str]) -> str:
    assertion = assertion.replace("return_value", "result").replace("assert", "return")
    if parser is not None:
        return get_eval_code_with_parser(assertion, signature, args, parser)
    else:
        return get_eval_code_with_io_pairs(assertion, signature, args)


async def run_code(code: str, num_of_tc: int) -> tuple[int, int, int, str]:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as tmp:
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
            try:
                proc.terminate()
                await proc.wait()
            except ProcessLookupError:
                pass
            return 0, 0, num_of_tc, "Timeout"
        if proc.stdout is None:
            return 0, 0, num_of_tc, "No stdout"
        stdout = await proc.stdout.read()
        stderr = await proc.stderr.read() if proc.stderr else b""
        if stderr:
            return 0, 0, num_of_tc, stderr.decode(errors="replace").strip()

        stdout_str = stdout.decode().strip()
        if not stdout_str:
            return 0, 0, num_of_tc, "No stdout"

        processed = stdout_str.split()

        if len(processed) != 3:
            return 0, 0, num_of_tc, f"Invalid stdout format: {stdout_str}"

        return int(processed[0]), int(processed[1]), int(processed[2]), "Success"


def get_total(io_pairs) -> int:
    if isinstance(io_pairs, list):
        return len(io_pairs)
    else:
        return len(json.loads(io_pairs)["inputs"])


async def evaluate_one_assertion(data: dict, evalplus_data: dict) -> EvaluationResult:
    assertion = data["postcondition_alone"]
    problem_idx = data["task_id"].split("/")[-1]
    problem = evalplus_data[problem_idx]

    # Check completeness
    eval_code = get_eval_code(
        assertion, problem["signature"], problem["input_output"], problem["parser"]
    )
    complete_total = get_total(problem["input_output"])
    if complete_total == 0:
        complete_total = 1
    (
        true_cnt_correct,
        false_cnt_correct,
        error_cnt_correct,
        msg_completeness,
    ) = await run_code(eval_code, complete_total)

    # Check soundness
    eval_code = get_eval_code(
        assertion,
        problem["signature"],
        problem["mutated_input_output"],
        problem["parser"],
    )
    sound_total = get_total(problem["mutated_input_output"])
    if sound_total == 0:
        sound_total = 1
    (
        true_cnt_mutated,
        false_cnt_mutated,
        error_cnt_mutated,
        msg_soundness,
    ) = await run_code(
        eval_code,
        sound_total,
    )

    return EvaluationResult(
        task_id=data["task_id"],
        assertion=assertion,
        is_complete=(
            false_cnt_correct == 0 and true_cnt_correct > 0 and error_cnt_correct == 0
        ),
        is_sound=false_cnt_mutated > true_cnt_mutated,
        complete_ratio=true_cnt_correct / complete_total,
        sound_ratio=false_cnt_mutated / sound_total,
        true_cnt_correct=true_cnt_correct,
        false_cnt_correct=false_cnt_correct,
        error_cnt_correct=error_cnt_correct,
        true_cnt_mutated=true_cnt_mutated,
        false_cnt_mutated=false_cnt_mutated,
        error_cnt_mutated=error_cnt_mutated,
        msg_completeness=msg_completeness,
        msg_soundness=msg_soundness,
    )


def aggregate_results(
    results: List[EvaluationResult], exp_name: str
) -> AggregatedResult:
    completeness_ratio = sum(result.is_complete for result in results) / len(results)
    soundness_ratio = sum(result.is_sound for result in results) / len(results)
    average_complete_ratio = sum(result.complete_ratio for result in results) / len(
        results
    )
    average_sound_ratio = sum(result.sound_ratio for result in results) / len(results)
    sound_and_complete = sum(
        result.is_complete and result.is_sound for result in results
    )
    complete_only = sum(
        result.is_complete and not result.is_sound for result in results
    )
    sound_only = sum(not result.is_complete and result.is_sound for result in results)
    failed = sum(not result.is_complete and not result.is_sound for result in results)
    return AggregatedResult(
        exp_name=exp_name,
        sound_and_complete=sound_and_complete,
        complete_only=complete_only,
        sound_only=sound_only,
        failed=failed,
        completeness_ratio=completeness_ratio,
        soundness_ratio=soundness_ratio,
        average_complete_ratio=average_complete_ratio,
        average_sound_ratio=average_sound_ratio,
    )


async def evaluate_target_directory(target_directory: str, dataset: str) -> List[EvaluationResult]:
    data = read_target_directory(target_directory)
    evalplus_data = load_dataset(dataset)
    tasks = [evaluate_one_assertion(d, evalplus_data) for d in data]
    return await atqdm.gather(*tasks)


@click.command()
@click.argument("target_directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--dataset",
    type=click.Path(exists=True),
    default=str(get_evalplus_dataset_file()),
    show_default=True,
)
@click.option("--exp_name", type=str, required=True)
def main(target_directory: str, dataset: str, exp_name: str):
    results = asyncio.run(evaluate_target_directory(target_directory, dataset))
    with open(os.path.join(target_directory, "evaluation_results.json"), "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=4)
    with open(os.path.join(target_directory, "aggregated_result.json"), "w") as f:
        agg_result = aggregate_results(results, exp_name)
        json.dump(agg_result.model_dump(), f, indent=4)
    print(agg_result)


if __name__ == "__main__":
    main()  # type: ignore[misc]
