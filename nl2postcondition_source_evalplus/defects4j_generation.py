import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from benchmarks import load_defects4j_method_examples
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, Model, get_model
from prompts import genOneWithRefJava

JAVA_SYSTEM_PROMPT = (
    "You are a programming assistant that generates executable Java only. "
    "You generate correct code, so you only generate code you are sure of. "
    "You have Java comments explaining your intent when possible."
)
PROMPT_TEMPLATES = genOneWithRefJava
TO_GENERATE_FULL = "symbolic postcondition"
TO_GENERATE_SHORT = "postcondition"
TO_GENERATE_GOAL = "means"
TO_GENERATE_SHORT_CAPS = "POSTCONDITION"


@dataclass(frozen=True)
class MethodRecord:
    id: str
    project: str
    bug_id: str
    method: str
    method_signature: str
    method_file_path: str
    javadoc: dict[str, Any]
    reference_code: str

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "MethodRecord":
        return cls(
            id=record["id"],
            project=record["project"],
            bug_id=record["bug_id"],
            method=record.get("method", record.get("method_name", "")),
            method_signature=record["method_signature"],
            method_file_path=record.get("file", ""),
            javadoc=record["javadoc"],
            reference_code=record["reference_code"],
        )


@dataclass(frozen=True)
class GenerationResult:
    id: str
    project: str
    bug_id: str
    method: str
    method_signature: str
    method_file_path: str
    prompt_version: str
    javadoc: dict[str, Any]
    reference_code: str
    prompt: str
    raw_response: str
    assertion: str


def build_prompt(method_record: MethodRecord, prompt_version: str) -> str:
    template = PROMPT_TEMPLATES[prompt_version]
    code_stub_and_docstring = "\n\n".join(
        [
            "/**",
            json.dumps(method_record.javadoc, indent=4, ensure_ascii=True),
            "*/",
            method_record.reference_code,
        ]
    )
    return template.substitute(
        codeStubAndDocstring=code_stub_and_docstring,
        toGenerateFull=TO_GENERATE_FULL,
        toGenerateShort=TO_GENERATE_SHORT,
        toGenerateGoal=TO_GENERATE_GOAL,
        toGenerateShortCaps=TO_GENERATE_SHORT_CAPS,
        promptAdds="",
        entrypoint=method_record.method_signature,
    )


def extract_code_block(text: str, language: str = "java") -> str:
    fence = f"```{language}"
    if fence in text:
        start = text.find(fence) + len(fence)
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()

    if "```" in text:
        first = text.find("```") + 3
        second = text.find("```", first)
        if second != -1:
            return text[first:second].strip()

    return text.strip()


def extract_java_assertion(text: str) -> str:
    code = extract_code_block(text, language="java")
    lines = [line.rstrip() for line in code.splitlines() if line.strip()]
    if not lines:
        return ""

    selected: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("assert "):
            selected.append(line)

    if selected:
        return "\n".join(selected)

    return "\n".join(lines)


async def generate_one(
    model: Model,
    method_record: MethodRecord,
    prompt_version: str,
    semaphore: asyncio.Semaphore,
) -> GenerationResult:
    prompt = build_prompt(method_record, prompt_version)
    messages = [
        ChatMessageSystem(content=JAVA_SYSTEM_PROMPT),
        ChatMessageUser(content=prompt),
    ]

    async with semaphore:
        result = await model.generate(input=messages)

    raw_response = result.message.content
    if isinstance(raw_response, list):
        raw_response = "\n".join(str(getattr(part, "text", part)) for part in raw_response)

    return GenerationResult(
        id=method_record.id,
        project=method_record.project,
        bug_id=method_record.bug_id,
        method=method_record.method,
        method_signature=method_record.method_signature,
        method_file_path=method_record.method_file_path,
        prompt_version=prompt_version,
        javadoc=method_record.javadoc,
        reference_code=method_record.reference_code,
        prompt=prompt,
        raw_response=str(raw_response),
        assertion=extract_java_assertion(str(raw_response)),
    )


def write_jsonl(path: Path, rows: list[GenerationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")


async def run_generation(
    model_name: str,
    dataset_path: str,
    output_dir: Path,
    limit: int | None,
    max_concurrency: int,
) -> None:
    benchmark_cfg = type(
        "BenchmarksCfg",
        (),
        {"location": dataset_path, "name": "defects4j"},
    )()
    method_records = load_defects4j_method_examples(benchmark_cfg, limit=limit)
    typed_method_records = [MethodRecord.from_dict(record) for record in method_records]
    model = get_model(model_name)
    semaphore = asyncio.Semaphore(max_concurrency)

    for prompt_version in ("simple", "base"):
        tasks = [
            generate_one(model, method_record, prompt_version, semaphore)
            for method_record in typed_method_records
        ]
        rows = await asyncio.gather(*tasks)
        write_jsonl(output_dir / f"{prompt_version}.jsonl", rows)
