-- Safe migration to add enable_pdf_upload column
-- This migration:
-- 1. Adds the column (will error if already exists - run manually)
-- 2. Sets a safe default value (false) for all existing rows
-- 3. Does NOT modify or delete any existing data

-- Add the column with a default value (SQLite syntax)
-- Note: SQLite doesn't support IF NOT EXISTS in ALTER TABLE ADD COLUMN
-- If column already exists, this will error - that's okay, just skip it
ALTER TABLE user_preferences
ADD COLUMN enable_pdf_upload BOOLEAN NOT NULL DEFAULT 0;

-- To run this migration:
-- sqlite3 data/app.db < migration_add_pdf_upload.sql
-- OR
-- sqlite3 data/app.db "ALTER TABLE user_preferences ADD COLUMN enable_pdf_upload BOOLEAN NOT NULL DEFAULT 0;"
