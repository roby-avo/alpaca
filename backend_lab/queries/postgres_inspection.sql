SELECT tablename
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;

SELECT table_name, ordinal_position, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name IN ('entities', 'entity_triples', 'query_cache', 'sample_entity_cache')
ORDER BY table_name, ordinal_position;

SELECT relname, n_live_tup
FROM pg_stat_user_tables
WHERE relname IN ('entities', 'entity_triples', 'query_cache', 'sample_entity_cache')
ORDER BY relname;

SELECT tablename, indexname
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename IN ('entities', 'entity_triples')
ORDER BY tablename, indexname;

SELECT qid, label, coarse_type, fine_type, item_category, popularity, prior
FROM entities
ORDER BY updated_at DESC
LIMIT 20;
