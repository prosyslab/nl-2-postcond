import json
import re
from pathlib import Path

from evalplus.data import get_human_eval_plus
from dataset_paths import get_defects4j_dataset_file


def load_evalplus_subset(evalplus_cfg):
    """
    Loads a subset of the humaneval+ benchmarks in the right format
    """
    problems = get_human_eval_plus()

    # if we are running all problems, return all problems
    if evalplus_cfg.run_all:
        return problems

    # otherwise, we are running a subset of the problems
    # see if we are running a specific subset or excluding a specific subset
    assert len(evalplus_cfg.run_only) > 0 or len(evalplus_cfg.run_except) > 0, (
        "If not running all problems, you must specify either a subset to run or a subset to exclude"
    )
    assert len(evalplus_cfg.run_only) == 0 or len(evalplus_cfg.run_except) == 0, (
        "If not running all problems, you must specify either a subset to run or a subset to exclude, not both"
    )

    filtered_problems = {}

    # if we are running a subset of the problems, filter those out
    for key, value in problems.items():
        stim_num = int(key[key.find("/") + 1 :])
        if len(evalplus_cfg.run_only) > 0 and stim_num in evalplus_cfg.run_only:
            filtered_problems[key] = value

        if len(evalplus_cfg.run_except) > 0 and stim_num not in evalplus_cfg.run_except:
            filtered_problems[key] = value

    return filtered_problems

METHOD_ID_SANITIZER = re.compile(r"[^0-9A-Za-z]+")


def sanitize_method_identifier(text: str) -> str:
    sanitized = METHOD_ID_SANITIZER.sub("_", text).strip("_")
    return sanitized or "method"


def extract_method_name(signature: str) -> str:
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", signature)
    if match:
        return match.group(1)
    parts = signature.strip().split()
    return parts[-1] if parts else "method"


def resolve_defects4j_dataset_path(benchmarks_cfg) -> Path:
    dataset_path = Path(
        getattr(benchmarks_cfg, "location", str(get_defects4j_dataset_file()))
    )
    if dataset_path.is_dir():
        dataset_path = dataset_path / "defects4j.json"
    return dataset_path


def load_expecto_defects4j_methods(bugs, limit=None):
    to_return = []
    ids_count = {}

    for bug in bugs:
        project = bug["project"]
        bug_id = str(bug["bug_id"])

        for method_dump in bug.get("method_dumps", []):
            method_info = method_dump["method_info"]
            method_signature = method_info["signature"]
            method_name = extract_method_name(method_signature)
            method_token = sanitize_method_identifier(
                f"{method_info['file']}_{method_signature}"
            )
            base_id = f"{project}_{bug_id}_{method_token}"
            ids_count[base_id] = ids_count.get(base_id, 0) + 1
            task_id = base_id
            if ids_count[base_id] > 1:
                task_id = f"{base_id}_{ids_count[base_id]}"

            to_return.append(
                {
                    "task_id": task_id,
                    "id": task_id,
                    "project": project,
                    "bug_id": bug_id,
                    "method_name": method_name,
                    "method_signature": method_signature,
                    "javadoc": method_info.get("javadoc", {}),
                    "reference_code": method_info.get("code", ""),
                    "file": method_info.get("file", ""),
                    "entry_schema": method_info.get("entry_schema", {}),
                    "exit_schema": method_info.get("exit_schema", {}),
                }
            )
            if limit is not None and len(to_return) >= limit:
                return to_return

    return to_return


def load_defects4j_method_examples(benchmarks_cfg, limit=None):
    dataset_path = resolve_defects4j_dataset_path(benchmarks_cfg)
    with open(dataset_path, "r") as f:
        loaded = json.load(f)

    bugs = loaded if isinstance(loaded, list) else [loaded]
    return load_expecto_defects4j_methods(bugs, limit=limit)


def load_defects4j(benchmarks_cfg):
    return load_defects4j_method_examples(benchmarks_cfg)


def load_benchmarks(benchmarks_cfg):
    """
    Loads the benchmark problems from the specified benchmark
    """

    # We are running with evalplus
    if benchmarks_cfg.name == "evalplus":
        # load all (or a subset) of the humaneval+ benchmark
        return load_evalplus_subset(benchmarks_cfg)
    elif benchmarks_cfg.name == "Defects4J" or benchmarks_cfg.name == "defects4j":
        return load_defects4j(benchmarks_cfg)
    elif "apps" in benchmarks_cfg.name:
        return load_apps(benchmarks_cfg)
    else:
        raise ValueError("Invalid benchmark name: {}".format(benchmarks_cfg.name))


def load_apps(apps_cfg):
    """
    Loads the apps benchmark
    """
    benchmark_path = apps_cfg.location
    with open(benchmark_path, "r") as f:
        apps_list = json.load(f)

    return {str(d["problem_id"]): d for d in apps_list}
