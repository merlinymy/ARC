-- Additive migration: per-conversation response stance.
--
-- Companion to migration_add_effort.sql. Adds the `mode` column that records
-- which stance (ask | brainstorm | develop | refine | critique | draft) a
-- conversation was last used in, so reopening a thread restores it.
--
-- This migration:
-- 1. Adds one column (will error if it already exists - run manually, skip on error)
-- 2. Backfills every existing row with 'ask', the default stance
-- 3. Does NOT modify or delete any existing column or row

-- SQLite has no ALTER TABLE ... ADD COLUMN IF NOT EXISTS; if the column is
-- already there this errors and that is fine, just skip it.
ALTER TABLE conversations
ADD COLUMN mode VARCHAR(20) NOT NULL DEFAULT 'ask';

-- To run this migration:
-- sqlite3 data/app.db < migration_add_conversation_mode.sql
-- OR
-- sqlite3 data/app.db "ALTER TABLE conversations ADD COLUMN mode VARCHAR(20) NOT NULL DEFAULT 'ask';"
