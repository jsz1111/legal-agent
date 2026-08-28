"""互殴场景 API 冒烟：清单合情理 + 时效横幅 + 聊天行出现 + 方案 Word 交付物。"""
from __future__ import annotations

import json
import sys
import urllib.request

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
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def checklist_labels(debug) -> list[dict]:
    return [
        {
            "label": row.get("label"),
            "collect_mode": row.get("collect_mode"),
            "decay_risk": row.get("decay_risk"),
            "status": row.get("status"),
            "active": row.get("active"),
        }
        for row in (debug or {}).get("evidence_checklist", [])
        if row.get("active")
    ]


def main() -> int:
    user = "smoke_brawl"
    ok = True

    r = chat(user, "s1", "我在小区门口被人打伤了，对方先动的手")
    debug = r.get("debug") or {}
    rows = checklist_labels(debug)
    labels = {row["label"] for row in rows}

    print("== 首次回复片段 ==")
    print((r.get("reply") or "")[:400].replace("\n", " "))

    print("\n== 证据清单（首次，陌生人互殴）==")
    for row in rows:
        print(f"  {row['label']} | mode={row['collect_mode']} decay={row['decay_risk']} {row['status']}")

    cctv = next((x for x in rows if "监控" in x["label"]), None)
    assert cctv, "监控行缺失"
    assert cctv["collect_mode"] == "retrieve", f"监控应 retrieve，实际 {cctv['collect_mode']}"
    assert cctv["decay_risk"] is True, "监控应 decay_risk"
    print("\n[PASS] 监控行 = retrieve + decay_risk")

    assert not any("聊天" in x["label"] for x in rows), "陌生互殴不应出现聊天记录行"
    assert not any("转账" in x["label"] for x in rows), "陌生互殴不应出现转账行"
    assert not any("通话" in x["label"] for x in rows), "陌生互殴不应出现通话行"
    print("[PASS] 无聊天/转账/通话行")

    # 统一时效横幅出现在收集回复里
    reply = r.get("reply") or ""
    if "易消失证据提示" in reply:
        print("[PASS] 收集回复含统一时效横幅")
    else:
        print("[INFO] 本次回复未含时效横幅（看 debug 判断是否在收集阶段）")

    # 用户补充“微信联系过” → 聊天行出现
    r2 = chat(user, "s1", "我们微信联系过")
    rows2 = checklist_labels(r2.get("debug") or {})
    chat_row = next((x for x in rows2 if "聊天" in x["label"]), None)
    assert chat_row, "回复“微信联系过”后聊天记录行未出现"
    print("\n[PASS] 回复“我们微信联系过”后聊天记录行出现:", chat_row)

    # 生成文书 → 方案 Word 版
    r3 = chat(user, "s1", "生成文书")
    doc = r3.get("document") or {}
    print("\n== 生成文书响应 ==")
    print("  doc_type:", doc.get("doc_type"))
    print("  filename:", doc.get("filename"))
    print("  official_template_note:", (doc.get("official_template_note") or "")[:80])
    assert doc.get("doc_type") == "维权行动方案（Word 版）", doc.get("doc_type")
    print("[PASS] 生成文书 → 方案 Word 版（不代填起诉状）")

    print("\n=== 冒烟全部通过 ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"\n[FAIL] {exc}")
        sys.exit(1)
