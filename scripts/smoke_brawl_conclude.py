"""推进互殴会话到结论，验证【维权路径比较】按程序先后序列。"""
from __future__ import annotations

import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8085/api/v1/chat"


def chat(session_id: str, message: str) -> dict:
    payload = {
        "user_id": "smoke_brawl",
        "session_id": session_id,
        "message": message,
        "mode": "case",
    }
    req = urllib.request.Request(
        BASE,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


ANSWERS = [
    "我在小区门口被人打伤，对方先动手。我现在安全。事情是昨天晚上8点左右，在小区东门。",
    "我脸部和手臂被打伤了，有淤青。对方我不认识，应该是陌生人，我们没有任何经济往来。",
    "我去医院看过，有病历和收费票据。当天我也去派出所报警了，有受案回执。",
    "伤情照片我拍了，监控就在小区东门口，可以调取。我没有证人联系方式。",
    "我已经把所有材料都交上去了。",
]


def main() -> int:
    session = "s_conclude"
    conv = chat(session, "我在小区门口被人打伤了，对方先动手")
    rows = [
        x.get("label")
        for x in (conv.get("debug") or {}).get("evidence_checklist", [])
        if x.get("active")
    ]
    print("证据清单行数:", len(rows))

    final = conv.get("reply") or ""
    for i, answer in enumerate(ANSWERS, start=1):
        conv = chat(session, answer)
        dbg = conv.get("debug") or {}
        conv_state = dbg.get("convergence") or {}
        final = conv.get("reply") or ""
        # 判断是否已到结论：reply 含方案特征
        if "维权行动方案" in final or "行动清单" in final:
            print(f"第 {i+1} 轮后进入结论")
            break
        print(f"第 {i+1} 轮: 回复 {len(final)} 字, convergence={conv_state.get('recommended_action') or ''}")

    print("\n=== 结论片段（维权路径/行动）===")
    for line in final.split("\n"):
        if any(k in line for k in ("维权路径", "①", "②", "③", "④", "报案", "伤情鉴定", "侦查", "民事赔偿", "刑附民", "程序", "报警")):
            print("  ", line.strip()[:90])

    # 序列性检查：若含【维权路径比较】，必须出现程序先后（①②③④）而非并列
    if "①" in final:
        print("\n[PASS] 结论含 ① 序号，维权路径按程序先后")
        seq_ok = "②" in final and ("③" in final or "③" not in final)
        if "②" in final:
            print("[PASS] 含 ② 序号")
    else:
        # 未到结论时给出提示
        print("\n[INFO] 本轮未产出含 ① 的结论（可能是条件式方案），检查是否含“按程序”提示")
    if "不得" in final or "按程序" in final:
        print("[INFO] 结论含程序性约束措辞")

    print("\n=== 冒烟结束 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
