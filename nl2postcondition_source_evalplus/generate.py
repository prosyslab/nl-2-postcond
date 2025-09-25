import asyncio

import click

EXP_CONFIGS = [
    "generateLLMSamplesSimple.yaml",
    "generateLLMSamplesBase.yaml",
    "generateLLMSamplesSimpleRefCode.yaml",
    "generateLLMSamplesBaseRefCode.yaml",
]

PPX_CONFIG = "preprocessSamples.yaml"
EVAL_CONFIG = "evaluation.yaml"


def log_cmd(args: list[str]):
    joined = " ".join(args)
    print(f"Executing {joined}")


async def run_exp(target_config: str, idx: int, benchmark: str):
    await asyncio.sleep(idx * 5)
    exp_name = target_config.replace("generateLLMSamples", "").replace(".yaml", "")
    args = [
        "python3",
        "llm_sample_generator.py",
        f"experiment={target_config}",
        f"benchmarks={benchmark}",
        f"hydra.run.dir=llm_gen_outputs/{exp_name}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}",
    ]
    log_cmd(args)
    result = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE)
    stdout, _ = await result.communicate()
    exit_code = result.returncode
    if exit_code != 0:
        print(f"\033[91mGeneration failed for {target_config}\033[0m")
        return
    saved_path = stdout.decode().split("\n")[-2]
    print(f"Saved path: {saved_path}")

    args = [
        "python3",
        "response_preprocessing.py",
        f"experiment={PPX_CONFIG}",
        f"experiment.samplesFolder={saved_path}",
        f"benchmarks={benchmark}",
        f"experiment.exp_name={exp_name}",
        f"hydra.run.dir=response_preprocess_outputs/{exp_name}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}",
    ]
    log_cmd(args)
    result = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE)
    stdout, _ = await result.communicate()
    exit_code = result.returncode
    if exit_code != 0:
        print(f"\033[91mResponse preprocessing failed for {target_config}\033[0m")
        return
    saved_path = stdout.decode().split("\n")[-2]
    print(f"Saved path: {saved_path}")

    args = [
        "python3",
        "evaluation.py",
        f"experiment={EVAL_CONFIG}",
        f"experiment.preprocessedFolder={saved_path}",
        f"experiment.exp_name={exp_name}",
        f"benchmarks={benchmark}",
        f"hydra.run.dir=evaluation_outputs/{exp_name}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}",
    ]
    log_cmd(args)
    proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE)
    stdout, _ = await proc.communicate()
    return_code = proc.returncode
    if return_code != 0:
        print(f"\033[91mEvaluation failed for {saved_path}\033[0m")
        return
    else:
        print(f"\033[92mEvaluation succeeded for {saved_path}\033[0m")
        print(stdout.decode())


async def run_all(benchmark):
    tasks = []
    for idx, config in enumerate(EXP_CONFIGS):
        tasks.append(run_exp(config, idx, benchmark))
    return await asyncio.gather(*tasks)


@click.command()
@click.option("--benchmark", type=str, required=True)
def main(benchmark):
    asyncio.run(run_all(benchmark))


if __name__ == "__main__":
    main()
