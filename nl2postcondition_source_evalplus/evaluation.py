import asyncio
import json
import os
import tempfile
from textwrap import indent

import log

# Optional dependency: tqdm.asyncio. Provide a minimal fallback if missing.
try:
    from tqdm.asyncio import tqdm as atqdm  # type: ignore

    _HAS_TQDM = True
except Exception:  # pragma: no cover
    _HAS_TQDM = False

    class _ATQDM:  # minimal async gather shim when tqdm is unavailable
        @staticmethod
        async def gather(*tasks):  # type: ignore[no-untyped-def]
            return await asyncio.gather(*tasks)

    atqdm = _ATQDM()  # type: ignore[assignment]

# Optional dependency: hydra/omegaconf. Allow import-time success for linters and
# provide graceful runtime errors if actually used without being installed.
try:
    import hydra  # type: ignore
    from hydra.core.config_store import ConfigStore  # type: ignore
    from hydra.utils import to_absolute_path  # type: ignore
    from omegaconf import DictConfig  # type: ignore
except Exception:  # pragma: no cover
    from types import SimpleNamespace

    def _noop_decorator(*args, **kwargs):  # type: ignore[no-untyped-def]
        def _wrap(fn):  # type: ignore[no-untyped-def]
            return fn

        return _wrap

    hydra = SimpleNamespace(main=_noop_decorator)  # type: ignore[assignment]

    class DictConfig:  # type: ignore[misc]
        pass

    class ConfigStore:  # type: ignore[misc]
        @staticmethod
        def instance():  # type: ignore[no-untyped-def]
            raise RuntimeError("Hydra is required to run this script.")

    def to_absolute_path(path: str) -> str:  # type: ignore[misc]
        return path


PPX_SAMPLE_JSONL = "preprocessed_samples.jsonl"


def load_dataset(dataset: str) -> dict:
    if "apps" in dataset:
        with open(dataset, "r") as f:
            d = json.load(f)
        for el in d:
            el["input_output"] = parse_arg(el["parser"], el["input_output"])
            el["mutated_input_output"] = parse_arg(
                el["parser"], el["mutated_input_output"]
            )
    else:
        with open(dataset, "r") as f:
            d = json.load(f)
    return {str(d["problem_id"]): d for d in d}


def parse_arg(parser: str, io_pairs: str) -> list:
    env = {}
    exec(parser, env)
    io_pairs_dict = json.loads(io_pairs)
    args = []
    for i, o in zip(io_pairs_dict["inputs"], io_pairs_dict["outputs"]):
        try:
            args.append(eval(f"parser({repr(i)}, {repr(o)})", env))
        except Exception:
            continue
    return args


def read_target_directory(target_directory: str) -> list[dict]:
    assert os.path.exists(os.path.join(target_directory, PPX_SAMPLE_JSONL)), (
        f"File {os.path.join(target_directory, PPX_SAMPLE_JSONL)} does not exist"
    )
    # Lazy import to avoid hard dependency at module import time
    import jsonlines  # type: ignore

    with jsonlines.open(os.path.join(target_directory, PPX_SAMPLE_JSONL)) as reader:
        return list(reader)


def get_eval_code_with_args(assertion: str, signature: str, args: list) -> str:
    signature = signature.split("\n")[0]
    assertion = assertion.replace("assert", "return")
    eval_code = f"""from typing import *
import math
import re
{signature}
{indent(assertion, " " * 4)}

true_cnt = 0
false_cnt = 0
"""
    for arg in args:
        arg_str = ", ".join(repr(e) for e in arg)
        eval_code += f"v = postcondition({arg_str})\n"
        eval_code += "if v:\n    true_cnt += 1\nelse:\n    false_cnt += 1\n"
    eval_code += "print(f'{true_cnt} {false_cnt}', flush=True)\n"
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
"""
    for i, o in zip(io_pairs_dict["inputs"], io_pairs_dict["outputs"]):
        input_args_str = ", ".join(repr(arg) for arg in i)
        eval_code += f"v = postcondition({input_args_str}, {repr(o)})\n"
        eval_code += "if v:\n    true_cnt += 1\nelse:\n    false_cnt += 1\n"
    eval_code += "print(f'{true_cnt} {false_cnt}', flush=True)\n"
    return eval_code


def get_eval_code(assertion: str, signature: str, args) -> str:
    if isinstance(args, list):
        return get_eval_code_with_args(assertion, signature, args)
    else:
        return get_eval_code_with_io_pairs(assertion, signature, args)


async def run_code(code: str) -> tuple[int, int]:
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
            return 0, 0
        if proc.stdout is None:
            return 0, 0
        stdout = await proc.stdout.read()
        stderr = await proc.stderr.read() if proc.stderr else b""
        if stderr:
            print(f"Stderr: {stderr}")
            return 0, 0

        stdout_str = stdout.decode().strip()
        if not stdout_str:
            return 0, 0

        processed = stdout_str.split()
        if len(processed) != 2:
            print(f"Warning: Unexpected output format: {processed}")
            return 0, 0

        return int(processed[0]), int(processed[1])


def get_total(io_pairs) -> int:
    if isinstance(io_pairs, list):
        return len(io_pairs)
    else:
        return len(json.loads(io_pairs)["inputs"])


async def evaluate_one_assertion(data: dict, evalplus_data: dict, logger) -> dict:
    assertion = data["postcondition_alone"]
    problem_idx = data["task_id"].split("/")[-1]
    problem = evalplus_data[problem_idx]

    # Check completeness
    eval_code = get_eval_code(assertion, problem["signature"], problem["input_output"])
    complete_total = get_total(problem["input_output"])
    if complete_total == 0:
        complete_total = 1
    true_cnt_correct, false_cnt_correct = await run_code(eval_code)

    # Check soundness
    eval_code = get_eval_code(
        assertion, problem["signature"], problem["mutated_input_output"]
    )
    sound_total = get_total(problem["mutated_input_output"])
    if sound_total == 0:
        sound_total = 1
    true_cnt_mutated, false_cnt_mutated = await run_code(eval_code)

    result = {
        "assertion": assertion,
        "is_complete": false_cnt_correct == 0 and true_cnt_correct > 0,
        "is_sound": false_cnt_mutated > true_cnt_mutated,
        "complete_ratio": true_cnt_correct / complete_total,
        "sound_ratio": false_cnt_mutated / sound_total,
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
    sound_and_complete = sum(
        result["is_complete"] and result["is_sound"] for result in results
    )
    complete_only = sum(
        result["is_complete"] and not result["is_sound"] for result in results
    )
    failed = sum(not result["is_complete"] for result in results)
    return {
        "exp_name": exp_name,
        "sound_and_complete": sound_and_complete,
        "complete_only": complete_only,
        "failed": failed,
        "completeness_ratio": completeness_ratio,
        "soundness_ratio": soundness_ratio,
        "average_complete_ratio": average_complete_ratio,
        "average_sound_ratio": average_sound_ratio,
    }


async def evaluate_target_directory(
    target_directory: str, dataset: str, logger
) -> list[dict]:
    data = read_target_directory(target_directory)
    evalplus_data = load_dataset(dataset)
    tasks = [evaluate_one_assertion(d, evalplus_data, logger) for d in data]
    return await atqdm.gather(*tasks)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):
    assert cfg.experiment
    assert cfg.benchmarks
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    # Resolve paths relative to original working directory (Hydra changes CWD)
    print_and_log, log_only = log.setup_output_dir(hydra_cfg)
    target_directory = to_absolute_path(cfg.experiment.preprocessedFolder)
    dataset = to_absolute_path(cfg.benchmarks.location)

    assert os.path.isdir(target_directory), f"Directory not found: {target_directory}"
    assert os.path.exists(dataset), f"Dataset not found: {dataset}"

    results = asyncio.run(
        evaluate_target_directory(target_directory, dataset, print_and_log)
    )
    with open(os.path.join(hydra_cfg.run.dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(hydra_cfg.run.dir, "aggregated_result.json"), "w") as f:
        agg_result = aggregate_results(results, cfg.experiment.exp_name)
        json.dump(agg_result, f, indent=4)
    print(agg_result)


if __name__ == "__main__":
    main()  # type: ignore[misc]
