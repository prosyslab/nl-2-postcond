import asyncio

EXP_CONFIGS = [
    "generateLLMSamplesSimple.yaml",
    "generateLLMSamplesBase.yaml",
    "generateLLMSamplesSimpleRefCode.yaml",
    "generateLLMSamplesBaseRefCode.yaml",
]

PPX_CONFIG = "preprocessSamples.yaml"


async def run_exp(target_config: str, idx: int):
    await asyncio.sleep(idx * 5)
    args = ["python3", "llm_sample_generator.py", f"experiment={target_config}"]
    result = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE)
    stdout, _ = await result.communicate()
    saved_path = stdout.decode().split("\n")[-2]
    print(f"Saved path: {saved_path}")

    args = [
        "python3",
        "response_preprocessing.py",
        f"experiment={PPX_CONFIG}",
        f"experiment.samplesFolder={saved_path}",
    ]
    result = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE)
    stdout, _ = await result.communicate()
    saved_path = stdout.decode().split("\n")[-2]
    print(f"Saved path: {saved_path}")

    args = [
        "python3",
        "evaluation.py",
        f"{saved_path}",
        "--dataset",
        "/data/duncan/expecto/benchmark/human_eval_plus.json",
        "--exp_name",
        target_config.replace("generateLLMSamples", "").replace(".yaml", ""),
    ]
    proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE)
    stdout, _ = await proc.communicate()
    return_code = proc.returncode
    if return_code != 0:
        print(f"\033[91mEvaluation failed for {saved_path}\033[0m")
    else:
        print(f"\033[92mEvaluation succeeded for {saved_path}\033[0m")
        print(stdout.decode())


async def main():
    tasks = []
    for idx, config in enumerate(EXP_CONFIGS):
        tasks.append(run_exp(config, idx))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
