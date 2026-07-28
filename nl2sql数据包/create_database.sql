\set ON_ERROR_STOP on

-- Run this file while connected to an administrative database, for example:
-- psql -d postgres -f create_database.sql
SELECT 'CREATE DATABASE legal_statistics_db ENCODING ''UTF8'''
WHERE NOT EXISTS (
    SELECT 1 FROM pg_database WHERE datname = 'legal_statistics_db'
)\gexec
