"""Run five isolated evidence-evaluation cases concurrently against the local API."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.legal_guide.state import GuideState
from src.infra.redis_cache import get_checkpointer_redis


MATERIAL_ROOT = ROOT / "test" / "materials" / "evidence_concurrency"


@dataclass(frozen=True)
class CaseSpec:
    group: str
    folder: str
    expected_domain_family: tuple[str, ...]


CASES = (
    CaseSpec("01_secondhand_phone", "01_secondhand_phone", ("cyber_data_fraud", "criminal_public_security")),
    CaseSpec("02_concert_ticket", "02_concert_ticket", ("contracts_property_housing",)),
    CaseSpec("03_overtime_wage", "03_overtime_wage", ("labor_social_security",)),
    CaseSpec("04_rental_deposit", "04_rental_deposit", ("contracts_property_housing",)),
    CaseSpec("05_traffic_injury", "05_traffic_injury", ("traffic_personal_injury",)),
)


def read_material(folder: str, pattern: str) -> str:
    paths = sorted((MATERIAL_ROOT / folder).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No material matched {folder}/{pattern}")
    return "\n\n".join(path.read_text(encoding="utf-8") for path in paths)


async def post_chat(
    client: httpx.AsyncClient,
    *,
    user_id: str,
    session_id: str,
    message: str,
    action: str = "message",
    target_case_id: str = "",
    regenerate_solution: bool = False,
) -> dict[str, Any]:
    response = await client.post(
        "/api/v1/chat",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "mode": "case",
            "action": action,
            "target_case_id": target_case_id,
            "regenerate_solution": regenerate_solution,
        },
    )
    response.raise_for_status()
    return response.json()


async def upload_document(
    client: httpx.AsyncClient,
    path: Path,
) -> dict[str, Any]:
    response = await client.post(
        "/api/v1/chat/upload-document",
        files={"file": (path.name, path.read_bytes(), "text/plain")},
    )
    response.raise_for_status()
    return response.json()


async def load_state(user_id: str, session_id: str) -> GuideState:
    redis = get_checkpointer_redis()
    raw = await redis.get(f"guide_state:{user_id}:{session_id}")
    if not raw:
        raise RuntimeError(f"Missing persisted state for {session_id}")
    return GuideState.model_validate_json(raw)


def basis_stats(checklist: list[dict[str, Any]]) -> dict[str, int]:
    refs = [
        ref
        for item in checklist
        for ref in (item.get("basis_refs") or [])
        if isinstance(ref, dict)
    ]
    return {
        "reference_count": len(refs),
        "with_body": sum(bool(str(ref.get("text") or "").strip()) for ref in refs),
        "with_url": sum(bool(str(ref.get("url") or "").strip()) for ref in refs),
    }


def state_summary(spec: CaseSpec, state: GuideState, elapsed: float) -> dict[str, Any]:
    coverage = state.evidence_coverage or {}
    checklist = [
        item
        for item in state.evidence_requirements
        if isinstance(item, dict) and item.get("active", True)
    ]
    uploaded_items = [
        item for item in state.evidence_items
        if item.get("availability") == "uploaded_copy"
    ]
    refs = basis_stats(checklist)
    own_marker = f"EVAL-{spec.group[:2]}"
    serialized_items = json.dumps(state.evidence_items, ensure_ascii=False)
    foreign_markers = [
        f"EVAL-{number:02d}"
        for number in range(1, 6)
        if f"EVAL-{number:02d}" != own_marker
        and f"EVAL-{number:02d}" in serialized_items
    ]
    statuses = {
        status: sum(item.get("status") == status for item in checklist)
        for status in (
            "preliminarily_covered",
            "partially_covered",
            "known_missing",
            "conflicted",
            "unresolved",
        )
    }
    checks = {
        "domain_reasonable": state.legal_domain in spec.expected_domain_family,
        "evaluation_version_incremented": state.evidence_evaluation_version >= 1,
        "uploaded_material_retained_in_state": bool(uploaded_items),
        "checklist_has_basis": refs["reference_count"] > 0,
        "basis_has_body_text": refs["with_body"] > 0,
        "basis_has_official_url": refs["with_url"] > 0,
        "no_cross_case_marker": not foreign_markers,
        "solution_includes_latest_evaluation": (
            state.solution_evidence_version == state.evidence_evaluation_version
        ),
    }
    return {
        "group": spec.group,
        "case_id": state.case_id,
        "domain": state.legal_domain,
        "elapsed_seconds": round(elapsed, 3),
        "evidence_evaluation_version": state.evidence_evaluation_version,
        "solution_version": state.solution_version,
        "solution_evidence_version": state.solution_evidence_version,
        "evidence_items": state.evidence_items,
        "coverage_counts": {
            "target_count": coverage.get("target_count", 0),
            "preliminarily_covered": coverage.get("preliminarily_covered_count", 0),
            "partially_covered": coverage.get("partial_count", 0),
            "known_missing": coverage.get("known_missing_count", 0),
            "unresolved": coverage.get("unresolved_count", 0),
        },
        "checklist_status_counts": statuses,
        "basis": refs,
        "foreign_markers": foreign_markers,
        "checks": checks,
        "passed": all(checks.values()),
    }


async def run_case(
    client: httpx.AsyncClient,
    *,
    user_id: str,
    session_id: str,
    spec: CaseSpec,
) -> dict[str, Any]:
    started = time.perf_counter()
    case_text = read_material(spec.folder, "case.txt")
    initial = await post_chat(
        client,
        user_id=user_id,
        session_id=session_id,
        message=(
            case_text
            + "\n\n这是并发测试，请先按现有信息生成一版方案，不再继续事实追问。"
        ),
    )

    evidence_paths = sorted((MATERIAL_ROOT / spec.folder).glob("evidence_*.txt"))
    uploads = await asyncio.gather(
        *(upload_document(client, path) for path in evidence_paths)
    )
    evidence_blocks = "\n\n".join(item["evidence_block"] for item in uploads)
    target_case_id = str((initial.get("debug") or {}).get("case_id") or "")
    await post_chat(
        client,
        user_id=user_id,
        session_id=session_id,
        message=(
            "以下是为本案提交的脱敏测试材料。请重新评估每项材料能证明什么、"
            "还缺什么，并把最新评估融入更新后的方案。\n\n"
            + evidence_blocks
        ),
        action="submit_evidence",
        target_case_id=target_case_id,
        regenerate_solution=True,
    )
    state = await load_state(user_id, session_id)
    return state_summary(spec, state, time.perf_counter() - started)


async def run(base_url: str, output: Path) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    user_id = f"codex-evidence-concurrency-{run_id}"
    session_ids = {
        spec.group: f"{spec.group}-{run_id}"
        for spec in CASES
    }
    wall_started = time.perf_counter()
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        timeout=None,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    ) as client:
        health = (await client.get("/health/deps")).json()
        results = await asyncio.gather(
            *(
                run_case(
                    client,
                    user_id=user_id,
                    session_id=session_ids[spec.group],
                    spec=spec,
                )
                for spec in CASES
            ),
            return_exceptions=True,
        )
    wall_elapsed = time.perf_counter() - wall_started
    cases: list[dict[str, Any]] = []
    for spec, result in zip(CASES, results):
        if isinstance(result, Exception):
            cases.append({
                "group": spec.group,
                "passed": False,
                "error": f"{type(result).__name__}: {result}",
            })
        else:
            cases.append(result)
    case_ids = [item.get("case_id") for item in cases if item.get("case_id")]
    report = {
        "run_id": run_id,
        "user_id": user_id,
        "base_url": base_url,
        "health": health,
        "wall_seconds": round(wall_elapsed, 3),
        "sum_case_seconds": round(sum(float(item.get("elapsed_seconds", 0)) for item in cases), 3),
        "parallelism_factor": round(
            sum(float(item.get("elapsed_seconds", 0)) for item in cases) / wall_elapsed,
            3,
        ) if wall_elapsed else 0,
        "session_ids": session_ids,
        "unique_case_ids": len(case_ids) == len(set(case_ids)) == len(CASES),
        "cases": cases,
        "passed": (
            len(cases) == len(CASES)
            and all(item.get("passed") for item in cases)
            and len(case_ids) == len(set(case_ids)) == len(CASES)
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8085")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "test" / "materials" / "reports" / "evidence_concurrency_report.json",
    )
    args = parser.parse_args()
    report = asyncio.run(run(args.base_url, args.output))
    concise = {
        "run_id": report["run_id"],
        "passed": report["passed"],
        "wall_seconds": report["wall_seconds"],
        "parallelism_factor": report["parallelism_factor"],
        "unique_case_ids": report["unique_case_ids"],
        "cases": [
            {
                "group": item.get("group"),
                "domain": item.get("domain"),
                "passed": item.get("passed"),
                "coverage_counts": item.get("coverage_counts"),
                "basis": item.get("basis"),
                "failed_checks": [
                    key for key, value in (item.get("checks") or {}).items()
                    if not value
                ],
                "error": item.get("error"),
            }
            for item in report["cases"]
        ],
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
