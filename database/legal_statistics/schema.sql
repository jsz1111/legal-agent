BEGIN;

CREATE SCHEMA IF NOT EXISTS legal_statistics;

CREATE TABLE IF NOT EXISTS legal_statistics.datasets (
    dataset_id TEXT PRIMARY KEY,
    yearbook_year SMALLINT NOT NULL,
    statistical_year_start SMALLINT NOT NULL,
    statistical_year_end SMALLINT NOT NULL,
    title TEXT NOT NULL,
    institution TEXT NOT NULL,
    topic TEXT NOT NULL,
    unit TEXT,
    source_file TEXT NOT NULL,
    source_sheet TEXT NOT NULL,
    source_url TEXT,
    source_sha256 CHAR(64) NOT NULL,
    source_rows INTEGER NOT NULL CHECK (source_rows >= 0),
    source_columns INTEGER NOT NULL CHECK (source_columns >= 0),
    data_start_row INTEGER,
    year_quality TEXT NOT NULL,
    extraction_status TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS legal_statistics.records (
    record_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES legal_statistics.datasets(dataset_id) ON DELETE CASCADE,
    source_row INTEGER NOT NULL CHECK (source_row > 0),
    row_label TEXT,
    row_data JSONB NOT NULL,
    search_text TEXT NOT NULL,
    UNIQUE (dataset_id, source_row)
);

CREATE TABLE IF NOT EXISTS legal_statistics.facts (
    fact_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES legal_statistics.datasets(dataset_id) ON DELETE CASCADE,
    record_id TEXT NOT NULL REFERENCES legal_statistics.records(record_id) ON DELETE CASCADE,
    yearbook_year SMALLINT NOT NULL,
    statistical_year_start SMALLINT NOT NULL,
    statistical_year_end SMALLINT NOT NULL,
    institution TEXT NOT NULL,
    topic TEXT NOT NULL,
    dimension_label TEXT,
    metric TEXT NOT NULL,
    metric_path TEXT NOT NULL,
    numeric_value NUMERIC,
    text_value TEXT NOT NULL,
    value_kind TEXT NOT NULL CHECK (value_kind IN ('number', 'percentage')),
    unit TEXT,
    source_row INTEGER NOT NULL CHECK (source_row > 0),
    source_column INTEGER NOT NULL CHECK (source_column > 0),
    quality_flag TEXT NOT NULL,
    UNIQUE (dataset_id, source_row, source_column)
);

CREATE TABLE IF NOT EXISTS legal_statistics.cells (
    dataset_id TEXT NOT NULL REFERENCES legal_statistics.datasets(dataset_id) ON DELETE CASCADE,
    source_row INTEGER NOT NULL CHECK (source_row > 0),
    source_column INTEGER NOT NULL CHECK (source_column > 0),
    cell_text TEXT,
    resolved_text TEXT,
    cell_type TEXT NOT NULL,
    is_merged BOOLEAN NOT NULL,
    merged_range TEXT,
    PRIMARY KEY (dataset_id, source_row, source_column)
);

CREATE TABLE IF NOT EXISTS legal_statistics.category_mappings (
    mapping_id BIGSERIAL PRIMARY KEY,
    source_label TEXT NOT NULL,
    canonical_label TEXT NOT NULL,
    valid_from_year SMALLINT,
    valid_to_year SMALLINT,
    mapping_scope TEXT NOT NULL DEFAULT 'case_category',
    notes TEXT,
    UNIQUE (source_label, canonical_label, valid_from_year, valid_to_year, mapping_scope)
);

CREATE INDEX IF NOT EXISTS ix_legal_stat_datasets_period
    ON legal_statistics.datasets(statistical_year_start, statistical_year_end);
CREATE INDEX IF NOT EXISTS ix_legal_stat_datasets_topic
    ON legal_statistics.datasets(topic, institution);
CREATE INDEX IF NOT EXISTS ix_legal_stat_records_dataset
    ON legal_statistics.records(dataset_id, source_row);
CREATE INDEX IF NOT EXISTS ix_legal_stat_records_search
    ON legal_statistics.records USING GIN (to_tsvector('simple', search_text));
CREATE INDEX IF NOT EXISTS ix_legal_stat_facts_period
    ON legal_statistics.facts(statistical_year_end, topic, metric);
CREATE INDEX IF NOT EXISTS ix_legal_stat_facts_dimension
    ON legal_statistics.facts(dimension_label);

COMMENT ON SCHEMA legal_statistics IS '中国法律年鉴2019-2021离线统计知识库';
COMMENT ON TABLE legal_statistics.datasets IS '每个年鉴工作表的数据集元数据与来源信息';
COMMENT ON TABLE legal_statistics.records IS '按原表行重建的JSON记录，适合AI检索和异构表读取';
COMMENT ON TABLE legal_statistics.facts IS '自动展平的数值事实长表，适合NL2SQL和ChatBI';
COMMENT ON TABLE legal_statistics.cells IS '原始单元格审计表，可追溯到文件、sheet、行列';

COMMIT;
