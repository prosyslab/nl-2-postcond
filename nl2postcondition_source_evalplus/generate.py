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
    last_line = stdout.decode().split("\n")[-2]

    args = [
        "python3",
        "response_preprocessing.py",
        f"experiment={PPX_CONFIG}",
        f"experiment.samplesFolder={last_line}",
    ]
    result = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE)
    stdout, _ = await result.communicate()
    last_line = stdout.decode().split("\n")[-2]


async def main():
    tasks = []
    for idx, config in enumerate(EXP_CONFIGS):
        tasks.append(run_exp(config, idx))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
