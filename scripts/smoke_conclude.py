"""Conclude-walk smoke: fragment self-containment + joint-analysis conclusion + no doc residue."""
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


def form(debug) -> dict:
    return (debug or {}).get("followup_form") or {}


def pick_answer(q: dict, idx: int) -> str:
    text = str(q.get("question") or "")
    # 用 idx==0 答一个碎片词"没有"来验证自包含入库
    if idx == 0:
        return "没有"
    if any(kw in text for kw in ("安全", "危险")):
        return "我现在安全"
    if any(kw in text for kw in ("时间", "何时", "什么时候")):
        return "昨天下午六点左右，在小区门口"
    if any(kw in text for kw in ("身份", "对方", "认识", "住址")):
        return "对方是隔壁单元的住户"
    if any(kw in text for kw in ("受伤", "伤情", "伤到", "医院")):
        return "我左手臂挫伤，去了医院拍片"
    if any(kw in text for kw in ("先动手", "动手", "起因", "冲突")):
        return "对方先动手，我先被推倒"
    if any(kw in text for kw in ("证据", "材料", "凭证", "记录")):
        return "有现场照片"
    return "不清楚"


def build_envelope(questions: list[dict]) -> str:
    lines = ["【动态追问表单回答】"]
    for idx, q in enumerate(questions, start=1):
        field_id = q.get("field_id") or ""
        if not field_id:
            continue
        lines.append(f"{idx} [{field_id}]")
        lines.append(f"回答: {pick_answer(q, idx - 1)}")
    return "\n".join(lines)


def main() -> int:
    user = "smoke_conclude"
    sid = user + "_" + uuid.uuid4().hex[:6]
    reply = ""
    detail_seen: list[str] = []
    converged = False

    for turn in range(1, 9):
        if turn == 1:
            msg = "我在小区门口被人打伤了，对方先动的手"
        else:
            f = form(debug)
            if f.get("questions"):
                msg = build_envelope(f.get("questions"))
                if turn >= 3:
                    msg += "\n按现有信息生成方案"
            else:
                msg = "按现有信息生成方案"

        resp = chat(user, sid, msg)
        reply = resp.get("reply") or ""
        debug = resp.get("debug") or {}
        detail = [f.get("statement") for f in debug.get("detail_store", []) if f.get("statement")]
        detail_seen = detail
        print(f"== TURN{turn} == {msg[:40].replace(chr(10),' ')}")
        print("  detail_store:", len(detail))
        ff = form(debug)
        print("  phase:", (debug.get('confidence_tier') or ''), "| form_kind:", ff.get("kind"), "| qs:", len(ff.get("questions") or []))

        if not ff.get("questions"):
            converged = True
            break

    print("\n=== 碎片词自包含检查 ===")
    anchored = [s for s in detail_seen if s and ("：" in s or ":" in s) and ("没有" in s or "无" in s)]
    for s in anchored:
        print("  *", s)
    if anchored:
        print("[PASS] 碎片答案已自包含入库:", anchored[0])
    else:
        print("[WARN] 未捕获到碎片词答案（本轮表单第一问答案未入库为'问题：没有'）")

    print("\n=== 结论检查 ===")
    print("converged:", converged)
    tail = reply[-600:]
    if "需要导出方案" in reply or "导出方案？" in reply:
        print("[FAIL] 结论仍含「📄 需要导出方案」残留")
    else:
        print("[PASS] 结论不含导出文书残留")
    if "遗漏" in reply and ("行动" in reply or "维权" in reply):
        print("[PASS] 结论含「遗漏的维权动作」")
    else:
        print("[WARN] 未检出'遗漏'词（可能未展开该栏目）")
    if "优势与劣势" in reply:
        print("[INFO] 含【优势与劣势】栏目")
    print("\n=== 结论摘要（末 800 字） ===")
    print(reply[-800:].replace("\n", " | "))

    with open("/tmp/smoke_conclude_reply.txt", "w", encoding="utf-8") as fh:
        fh.write(reply)
    print("\n[INFO] 完整结论已写入 /tmp/smoke_conclude_reply.txt")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"\n[FAIL] {exc}")
        sys.exit(1)
