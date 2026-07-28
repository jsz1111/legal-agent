# 案例库数据库接入

案例数据使用现有 PostgreSQL `legal_cases` 表，并通过增量字段保存完整元数据。数据库结构补丁位于 `schema.sql`，幂等导入和 Milvus 索引重建由 `scripts/import_case_pilot.py` 完成。

核心约定：

- PostgreSQL 是案例全文和元数据的事实来源。
- Milvus 只保存 `retrieval_text` 和向量，不保存整篇裁判文书。
- `case_id` 是跨 CSV、JSONL、SQLite 和 PostgreSQL 的稳定标识。
- Milvus 主键使用 PostgreSQL 数字主键，以便召回后批量补充详情。
- 新数据使用内部粗分类 `civil_case`，不修改原始案由，也不映射到某个 Agent 的维权领域。
- 数据包的商业“来源”字段不导入；原始裁判文书链接保留。

导入器会先创建临时 Milvus 集合，全部写入并校验数量后再切换为正式 `case_index`。旧集合会改名保留，便于人工确认后回滚或删除。
