"""
THE entry point: transcript + detections -> the artifact surrogate generation reads.

PURPOSE
    The production command. Reads one or more transcripts plus their detection
    files, runs the pipeline, and writes a validated artifact per transcript to
    `out/graphs/`.

FIT
    A thin CLI over `graph/loader.py` -> `graph/pipeline.py` ->
    `graph/serialize.py`. `--demo` borrows `demo.cases.detections_from_gold` so the
    handoff can be smoke-tested before a real detector exists.

HOW
    `main` is argument parsing and a loop; `build_one` is the actual work for one
    transcript. Failures are per-transcript -- a bad input file is reported and the
    loop continues -- and the process EXIT CODE reports the overall verdict (see
    the codes below), so a caller in a pipeline can branch on "ready" versus
    "needs review" without parsing stdout.

    ./venv/bin/python3 scripts/build_graph.py                    # every transcript in data/
    ./venv/bin/python3 scripts/build_graph.py interview_001      # just one
    ./venv/bin/python3 scripts/build_graph.py --demo             # the bundled samples
    KG_USE_LLM=1 ./venv/bin/python3 scripts/build_graph.py       # with the second line

Until now nothing in this repo wrote a graph to disk. `graph/serialize.py` existed and
was never called, so the only way to see a pipeline result was to render one of the
HTML reports -- which is a review surface, not a handoff.

Input layout (the contract `graph/loader.py` enforces):

    data/
    |-- transcripts/interview_001.txt
    |-- detections/interview_001.json      <- from the detection stage
    `-- metadata.json                      <- {"interview_001": {"interview_date": ...}}

Output: out/graphs/<transcript_id>.json, validated before it is written.

EXIT CODES matter here, because "blocking" is a review gate and not a warning:

    0   written, nothing blocking -- ready for surrogate generation
    2   written, but N fields need a human before surrogates are minted
    1   a transcript failed to process

`--demo` swaps the detection stage for the simulated perfect detector in
`samples/gold/*.json`, so the whole handoff can be smoke-tested before a real detector
exists. It is a test fixture, not a detector; never point it at real data.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent
for p in (str(REPO_ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from graph.loader import (Violation, load_stage_inputs, load_transcript,
                          make_mentions, resolve_overlaps)
from graph.pipeline import run_pipeline
from graph.serialize import serialize

SAMPLES = REPO_ROOT / "samples"
DEFAULT_GAZ = REPO_ROOT / "data" / "gazetteer.csv"


def _demo_inputs(tid: str):
    """Transcript + mentions from the gold annotations (a stand-in detector).

    Returns the same `(text, mentions, metadata)` triple `loader.load_stage_inputs`
    does, so `build_one` needs no other branch. A FIXTURE, not a detector -- never
    point `--demo` at real data.
    """
    from demo.cases import detections_from_gold

    text = load_transcript(SAMPLES / "transcripts" / f"{tid}.txt")
    gold = json.loads((SAMPLES / "gold" / f"{tid}.json").read_text(encoding="utf-8"))
    mentions = make_mentions(tid, resolve_overlaps(detections_from_gold(text, gold)))
    return text, mentions, {"interview_date": gold.get("interview_date")}


def _ids(args) -> list[str]:
    """Which transcript ids to process.

    Explicit ids on the command line win; otherwise every `.txt` in the relevant
    transcripts directory, sorted so runs are reproducible. A missing directory
    raises `Violation`, which `main` turns into exit code 1.
    """
    if args.transcript_ids:
        return list(args.transcript_ids)
    root = (SAMPLES if args.demo else Path(args.data_dir)) / "transcripts"
    if not root.is_dir():
        raise Violation(f"no transcripts directory at {root}")
    return sorted(p.stem for p in root.glob("*.txt"))


def build_one(tid: str, args) -> tuple[Path, list]:
    """Process one transcript end to end. Returns `(written path, blocking rows)`.

    Load (real detections or the demo fixture) -> run the pipeline -> serialize.
    `serialize` validates before writing, so a returned path means the artifact on
    disk is well-formed and its offsets agree with the transcript.
    """
    text, mentions, metadata = (
        _demo_inputs(tid) if args.demo
        else load_stage_inputs(tid, args.data_dir))
    entities, edges, info = run_pipeline(
        tid, text, mentions, metadata=metadata,
        gazetteer_path=str(args.gazetteer), run_coref=not args.no_coref)
    return serialize(tid, text, entities, edges, info, args.out), info["blocking"]


def _report(tid: str, path: Path, blocking: list) -> None:
    """Print one transcript's outcome, listing any blocking fields for a reviewer.

    Paths are shown relative to the working directory when possible, and each
    blocking reason is truncated so a long explanation does not swamp the summary.
    """
    print(f"  {tid}: wrote {path.relative_to(Path.cwd())
                            if path.is_relative_to(Path.cwd()) else path}")
    if not blocking:
        print("      no blocking fields -- ready for surrogate generation")
        return
    print(f"      {len(blocking)} BLOCKING field(s) -- a human decides before "
          f"surrogates are minted:")
    for eid, field, reason in blocking:
        print(f"        {eid}  {field}")
        print(f"          {reason[:150]}")


def main() -> int:
    """Parse arguments, build every requested transcript, and return the exit code.

    Exit codes are the interface (see the module docstring): 0 = written and
    nothing blocking, 2 = written but a human must resolve N fields first, 1 = at
    least one transcript failed. A failure is caught per transcript so one bad
    input does not abandon the rest of the batch.
    """
    ap = argparse.ArgumentParser(
        description="Build the knowledge-graph artifact for one or more transcripts.")
    ap.add_argument("transcript_ids", nargs="*",
                    help="transcript ids; default is every transcript found")
    ap.add_argument("--data-dir", default="data",
                    help="directory holding transcripts/, detections/, metadata.json")
    ap.add_argument("--out", default="out/graphs", help="where to write the artifacts")
    ap.add_argument("--gazetteer", default=str(DEFAULT_GAZ))
    ap.add_argument("--no-coref", action="store_true",
                    help="skip the coreference stage (faster; loads no model)")
    ap.add_argument("--demo", action="store_true",
                    help="use samples/ and the SIMULATED detector in samples/gold "
                         "(a fixture for smoke-testing, never real data)")
    args = ap.parse_args()

    if os.environ.get("KG_USE_LLM") == "1":
        print("LLM second line: ON")

    try:
        tids = _ids(args)
    except Violation as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    failed, blocked = [], 0
    for tid in tids:
        try:
            path, blocking = build_one(tid, args)
        except Violation as exc:
            print(f"  {tid}: FAILED -- {exc}", file=sys.stderr)
            failed.append(tid)
            continue
        except FileNotFoundError as exc:
            print(f"  {tid}: FAILED -- missing input: {exc}", file=sys.stderr)
            failed.append(tid)
            continue
        _report(tid, path, blocking)
        blocked += len(blocking)

    print()
    if failed:
        print(f"{len(failed)} transcript(s) failed: {', '.join(failed)}")
        return 1
    if blocked:
        print(f"{len(tids)} artifact(s) written, {blocked} blocking field(s) "
              f"outstanding -- resolve them before minting surrogates.")
        return 2
    print(f"{len(tids)} artifact(s) written, nothing blocking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
