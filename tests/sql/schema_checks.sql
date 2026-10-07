-- Run against a freshly migrated database: psql -v ON_ERROR_STOP=1 -f tests/sql/schema_checks.sql
-- Verifies (1) point-in-time reads ignore later restatements, (2) append-only tables reject mutation.
BEGIN;
INSERT INTO securities (security_id, asset_class, name) VALUES (1, 'equity', 'TestCo');
-- Original FY2024 revenue filed 2025-02-10, restated 2025-08-01
INSERT INTO fundamentals (security_id, metric, period_type, period_end, value, known_at)
VALUES (1,'revenue','FY','2024-12-31',100.0,'2025-02-10'),
       (1,'revenue','FY','2024-12-31', 92.0,'2025-08-01');

DO $$
DECLARE v_before double precision; v_after double precision;
BEGIN
  SELECT value INTO v_before FROM fundamentals
   WHERE security_id=1 AND metric='revenue' AND period_end='2024-12-31' AND known_at <= '2025-06-30'
   ORDER BY known_at DESC LIMIT 1;
  SELECT value INTO v_after FROM fundamentals
   WHERE security_id=1 AND metric='revenue' AND period_end='2024-12-31' AND known_at <= '2025-12-31'
   ORDER BY known_at DESC LIMIT 1;
  IF v_before <> 100.0 OR v_after <> 92.0 THEN
    RAISE EXCEPTION 'PIT read wrong: before=% after=%', v_before, v_after;
  END IF;
END $$;

INSERT INTO research_cycles (cycle_id, security_id, trigger_event, data_snapshot_at)
VALUES ('00000000-0000-0000-0000-000000000001', 1, 'initiation', now());
INSERT INTO decisions (security_id, cycle_id, recommendation, confidence, price_at_decision,
                       scenario_values, assumptions, risks, gate_results, disagreements,
                       judge_rationale, models_used, data_snapshot_at)
VALUES (1,'00000000-0000-0000-0000-000000000001','WATCH',0.5,10,'{}','{}','{}','{}','{}','test','[]',now());

DO $$
BEGIN
  BEGIN
    UPDATE decisions SET confidence = 0.9;
    RAISE EXCEPTION 'decisions UPDATE was allowed';
  EXCEPTION WHEN raise_exception THEN
    IF SQLERRM NOT LIKE '%append-only%' THEN RAISE; END IF;
  END;
  BEGIN
    DELETE FROM decisions;
    RAISE EXCEPTION 'decisions DELETE was allowed';
  EXCEPTION WHEN raise_exception THEN
    IF SQLERRM NOT LIKE '%append-only%' THEN RAISE; END IF;
  END;
END $$;
SELECT 'SCHEMA_CHECKS_PASSED' AS result;
ROLLBACK;
