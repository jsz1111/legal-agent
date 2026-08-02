# 测试目录

本目录统一保存项目测试代码、验收执行器和脱敏测试物料，不在项目根目录或 `scripts/` 中保留临时测试脚本。

- `test_*.py`：pytest 自动收集的单元、接口、工作流和集成回归测试；
- `evaluate_guide_scenarios.py`：真实环境多场景验收执行器；
- `run_evidence_concurrency.py`：五案件并发证据评估执行器；
- `materials/evidence_concurrency/`：依据公开案例脱敏重构的五组测试输入；
- `materials/images/`：多模态演示与图片解析测试物料；
- `materials/reports/`：并发测试报告。

常规回归：

```powershell
python -m pytest test -q
```

测试材料不是真实当事人的原始证据，也不能作为知识库法律依据。
