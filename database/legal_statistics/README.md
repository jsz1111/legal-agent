# 法律统计独立数据库

本目录定义独立 PostgreSQL 数据库 `legal_statistics_db`。它与项目现有
`legal_db` 隔离，统计数据不会混入法条、案例和用户咨询表。

## 年份口径

- `yearbook_year`：年鉴版本年份，当前为 2019、2020、2021。
- `statistical_year_start/end`：表格标题中的真实统计期间，主要为
  2018、2019、2020；没有显式年份时按 `yearbook_year - 1` 推断并标记。

## 表用途

- `datasets`：数据集目录、来源、年份、单位和提取状态。
- `records`：保持每张异构表的行级 JSON 表达，适合 AI 检索。
- `facts`：数值长表，供 NL2SQL、聚合和 ChatBI 使用。
- `cells`：原始单元格审计层，用于复核自动展平结果。
- `category_mappings`：跨年度统计口径映射，由人工审核后维护。

## 离线包内容

- `csv/`：PostgreSQL COPY 使用的 UTF-8 数据文件。
- `jsonl/`：其他 AI、RAG 或批处理工具可逐条读取的数据文件。
- `schema.sql`：独立 schema 和索引定义。
- `load_postgres.py`：显式指定数据库地址后执行导入。
- `validate_package.py`：不连接数据库即可完成完整性校验。
- `manifest.json`：数据量、年份、主题和提取错误汇总。

## 构建离线包

```powershell
python scripts/build_legal_statistics_package.py `
  --source-root "D:\BaiduNetdiskDownload\免费共享-中国法律年鉴 2020-2021" `
  --output-dir "data\processed\legal_statistics_2019_2021"
```

构建阶段只读取 Excel 并生成文件，不连接 PostgreSQL。

构建后可先离线验证：

```powershell
python validate_package.py --package-dir . --report validation_report.json
```

## 将来导入独立数据库

```powershell
psql -d postgres -f create_database.sql
python load_postgres.py `
  --package-dir . `
  --database-url "postgresql://legal:***@127.0.0.1:5433/legal_statistics_db"
```

首次自动抽取的事实需要结合 `quality_flag` 和原始 `cells` 表进行抽样复核。
