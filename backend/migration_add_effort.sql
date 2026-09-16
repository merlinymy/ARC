-- Safe migration to add the `effort` column that replaces `temperature`
-- This migration:
-- 1. Adds the column (will error if already exists - run manually)
-- 2. Sets a safe default value ('high') for all existing rows
-- 3. Does NOT modify or delete the existing `temperature` column or any other data
--    (temperature is left in place for historical rows; the app stops reading it)

-- Add the column with a default value (SQLite syntax)
-- Note: SQLite doesn't support IF NOT EXISTS in ALTER TABLE ADD COLUMN
-- If column already exists, this will error - that's okay, just skip it
ALTER TABLE user_preferences
ADD COLUMN effort VARCHAR(10) NOT NULL DEFAULT 'high';

-- To run this migration:
-- sqlite3 data/app.db < migration_add_effort.sql
-- OR
-- sqlite3 data/app.db "ALTER TABLE user_preferences ADD COLUMN effort VARCHAR(10) NOT NULL DEFAULT 'high';"
