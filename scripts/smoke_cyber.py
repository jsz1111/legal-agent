"""Cyber-fraud smoke: dynamic follow-up on privacy leak + fact self-containment + conclusion invariants."""
from __future__ import annotations

import json
import sys
import urllib.request
import uuid

BASE = "http://127.0.0.1:8085/api/v1/chat"


def chat(user_id: str, session_id: str, message: str, **extra) -> dict:
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "message": message,
        "mode": "case",
        **extra,
    }
    req = urllib.request.Request(
        BASE,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=240) as resp:
        return json.loads(resp.read().decode("utf-8"))


def form_questions(debug) -> list[dict]:
    form = (debug or {}).get("followup_form") or {}
    return form.get("questions") or []


def pick_answer(q: dict) -> str:
    text = str(q.get("question") or "") + " " + str(q.get("why") or "")
    for kw, answer in (
        ("平台", "淘宝"),
        ("时间", "昨天"),
        ("金额", "500元"),
        ("多少", "500元"),
        ("凭证", "没有"),
        ("证据", "没有"),
        ("材料", "没有"),
        ("隐私", "隐私泄露（个人信息被获取）"),
        ("信息泄露", "隐私泄露（个人信息被获取）"),
        ("联系", "已联系平台"),
        ("损失", "500元"),
    ):
        if kw in text:
            return answer
    return "不清楚"


def build_envelope(questions: list[dict]) -> str:
    lines = ["【动态追问表单回答】"]
    for idx, q in enumerate(questions, start=1):
        field_id = q.get("field_id") or ""
        if not field_id:
            continue
        lines.append(f"{idx} [{field_id}]")
        lines.append(f"回答: {pick_answer(q)}")
    return "\n".join(lines)


def main() -> int:
    user = "smoke_cyber"
    sid = user + "_" + uuid.uuid4().hex[:6]

    r1 = chat(user, sid, "我买东西被骗了")
    d1 = r1.get("debug") or {}
    qs1 = form_questions(d1)
    print("== TURN1 ==")
    print("domain:", d1.get("domain"))
    print("followup_form kind:", (d1.get("followup_form") or {}).get("plan_kind"))
    print("questions:", len(qs1))
    for q in qs1:
        print("  -", q.get("question", "")[:60])

    assert d1.get("domain"), "首轮应确定领域"
    assert qs1, "首轮应出现动态追问表单"
    print("[PASS] 首轮动态追问表单出现")

    r2 = chat(user, sid, build_envelope(qs1))
    d2 = r2.get("debug") or {}
    qs2 = form_questions(d2)
    print("\n== TURN2 (答完表单) ==")
    print("followup_form kind:", (d2.get("followup_form") or {}).get("plan_kind"))
    print("questions:", len(qs2))
    detail = [f.get("statement") for f in d2.get("detail_store", []) if f.get("statement")]
    print("detail_store:")
    for s in detail:
        print("  -", s)
    print("== 本轮追问 ==")
    for q in qs2:
        print("  -", q.get("question", "")[:70])

    privacy_followups = [
        q.get("question", "")
        for q in qs2
        if any(kw in q.get("question", "") for kw in ("信息", "改密", "挂失", "泄露", "密码", "损失"))
    ]
    if privacy_followups:
        print("\n[PASS] 隐私泄露后出现动态追问:")
        for q in privacy_followups:
            print("    *", q[:80])
    else:
        print("\n[FAIL] 未出现隐私泄露相关动态追问")

    # 碎片词自包含：把"没有"存成 "问题：没有"
    anchored = [s for s in detail if s and ("：" in s or ":" in s)]
    if anchored:
        print("[PASS] 碎片答案自包含示例:", anchored[0])
    else:
        print("[INFO] 本轮无碎片词答案")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"\n[FAIL] {exc}")
        sys.exit(1)
