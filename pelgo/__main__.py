"""CLI entry point: python -m pelgo --resume path/to/resume.pdf --jd "..." """
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from .agent import AgentRunner
from .pdf_extractor import parse_resume_pdf, parse_resume_text

_BATCH_CONCURRENCY = 5


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pelgo Career Intelligence Agent")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--resume", metavar="PATH", help="Path to a single candidate resume PDF")
    group.add_argument("--resume-text", metavar="TEXT", help="Raw resume text")
    group.add_argument("--resumes-dir", metavar="DIR", help="Folder of resume PDFs to process in batch")
    p.add_argument("--jd", required=True, metavar="TEXT_OR_URL", help="Job description or URL")
    p.add_argument("--job-id", metavar="UUID", help="Optional stable job ID (single-resume mode only)")
    p.add_argument("--output", metavar="PATH", help="Write JSON result to file (single) or directory (batch)")
    return p.parse_args()


async def _run_one(runner: AgentRunner, pdf_path: Path, jd: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        try:
            profile = parse_resume_pdf(str(pdf_path))
            print(f"  [{pdf_path.name}] {profile.name} | {len(profile.skills)} skills | "
                  f"{profile.years_experience}y | {profile.seniority_level}")
            result = await runner.run(profile, jd)
            return {"file": pdf_path.name, "ok": True, "result": json.loads(result.model_dump_json())}
        except Exception as exc:
            print(f"  [{pdf_path.name}] ERROR: {exc}", file=sys.stderr)
            return {"file": pdf_path.name, "ok": False, "error": str(exc)}


async def _run_batch(args: argparse.Namespace) -> None:
    folder = Path(args.resumes_dir)
    if not folder.is_dir():
        print(f"ERROR: not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"ERROR: no PDF files found in {folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdfs)} resumes — processing with concurrency={_BATCH_CONCURRENCY}")

    output_dir = Path(args.output) if args.output else folder / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = AgentRunner()
    sem = asyncio.Semaphore(_BATCH_CONCURRENCY)

    tasks = [_run_one(runner, pdf, args.jd, sem) for pdf in pdfs]
    results = await asyncio.gather(*tasks)

    ok = sum(1 for r in results if r["ok"])
    for entry in results:
        dest = output_dir / (Path(entry["file"]).stem + ".json")
        dest.write_text(json.dumps(entry, indent=2))

    print(f"\nDone: {ok}/{len(pdfs)} succeeded — results in {output_dir}")


async def _run(args: argparse.Namespace) -> None:
    if args.resume:
        path = Path(args.resume)
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        profile = parse_resume_pdf(str(path))
    else:
        profile = parse_resume_text(args.resume_text)

    print(f"Candidate: {profile.name} | Skills: {len(profile.skills)} | "
          f"Experience: {profile.years_experience}y | Seniority: {profile.seniority_level}")

    runner = AgentRunner()
    result = await runner.run(profile, args.jd, job_id=args.job_id)

    output = result.model_dump_json(indent=2)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Result written to {args.output}")
    else:
        print(output)


def main() -> None:
    args = _parse_args()
    if args.resumes_dir:
        asyncio.run(_run_batch(args))
    else:
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
