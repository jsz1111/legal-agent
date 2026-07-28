-- Additive schema for the reusable legal case library.
-- Existing legal_cases rows remain valid because every new column is nullable.

ALTER TABLE legal_cases ADD COLUMN IF NOT EXISTS case_id VARCHAR(64);
ALTER TABLE legal_cases ADD COLUMN IF NOT EXISTS original_url VARCHAR(1000);
ALTER TABLE legal_cases ADD COLUMN IF NOT EXISTS case_number VARCHAR(200);
ALTER TABLE legal_cases ADD COLUMN IF NOT EXISTS region VARCHAR(100);
ALTER TABLE legal_cases ADD COLUMN IF NOT EXISTS case_type VARCHAR(50);
ALTER TABLE legal_cases ADD COLUMN IF NOT EXISTS case_type_code VARCHAR(20);
ALTER TABLE legal_cases ADD COLUMN IF NOT EXISTS procedure VARCHAR(50);
ALTER TABLE legal_cases ADD COLUMN IF NOT EXISTS judgment_date VARCHAR(20);
ALTER TABLE legal_cases ADD COLUMN IF NOT EXISTS publication_date VARCHAR(20);
ALTER TABLE legal_cases ADD COLUMN IF NOT EXISTS parties TEXT;
ALTER TABLE legal_cases ADD COLUMN IF NOT EXISTS legal_basis TEXT;
ALTER TABLE legal_cases ADD COLUMN IF NOT EXISTS full_text TEXT;
ALTER TABLE legal_cases ADD COLUMN IF NOT EXISTS full_text_length INTEGER;
ALTER TABLE legal_cases ADD COLUMN IF NOT EXISTS retrieval_text TEXT;
ALTER TABLE legal_cases ADD COLUMN IF NOT EXISTS selection_tags TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS uq_legal_cases_case_id ON legal_cases(case_id);
CREATE INDEX IF NOT EXISTS ix_legal_cases_case_number ON legal_cases(case_number);
CREATE INDEX IF NOT EXISTS ix_legal_cases_region ON legal_cases(region);
CREATE INDEX IF NOT EXISTS ix_legal_cases_procedure ON legal_cases(procedure);
CREATE INDEX IF NOT EXISTS ix_legal_cases_judgment_date ON legal_cases(judgment_date);
