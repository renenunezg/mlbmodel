"""PostgreSQL acceptance for atomic forecasts and retryable evaluation."""
import json
import os
import subprocess
import uuid
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError


@pytest.mark.database
def test_publication_and_evaluation_recovery(monkeypatch):
    value = os.getenv("MLB_TEST_DATABASE_URL")
    if not value:
        pytest.skip("requires a disposable local PostgreSQL admin database")
    url = make_url(value)
    assert url.host in ("127.0.0.1", "localhost")
    name = "mlb_acceptance_" + uuid.uuid4().hex
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    local_url = url.set(database=name)
    db = create_engine(local_url, connect_args={"options": "-csearch_path=mlb,public"})
    try:
        with db.begin() as conn:
            conn.execute(text("""CREATE FUNCTION public.site_revalidate() RETURNS trigger
                LANGUAGE plpgsql AS $$ BEGIN RETURN NULL; END $$"""))
        root = Path(__file__).parents[1]
        for path in (root / "backend/sql/baseline/2026_10_05.sql",
                     root / "backend/sql/interval_coverage_population.sql",
                     root / "backend/sql/live_evaluation_reconciliation.sql",
                     root / "backend/sql/20261007171918_mlb_live_evaluation_completion.sql",
                     root / "backend/sql/history_record_bounds.sql"):
            subprocess.run(["psql", "-X", local_url.render_as_string(hide_password=False),
                            "-v", "ON_ERROR_STOP=1", "-f", str(path)],
                           check=True, capture_output=True)
        with db.begin() as conn:
            # Exercise the aggregate contract independently of the ledger's grading SQL.
            conn.execute(text("ALTER VIEW mlb.bet_ledger_agg_v RENAME TO acceptance_original_ledger"))
            conn.execute(text("""CREATE VIEW mlb.bet_ledger_agg_v AS
              SELECT * FROM (VALUES
                ('2026-08-01'::date, 'BOS', 'ml', true, false),
                ('2026-08-02'::date, 'BOS', 'ml', false, false),
                ('2026-08-03'::date, 'BOS', 'ml', true, false),
                ('2026-08-02'::date, 'NYY', 'ml', true, false)
              ) AS t(date, team, bet_type, won, push)"""))
            rows = conn.execute(text("SELECT * FROM mlb.bet_record_summary('2026-08-01','BOS','2026-08-02')")).all()
            assert rows == [('ml', 1, 1, 0)]
            assert conn.execute(text("SELECT * FROM mlb.bet_record_summary('2026-08-04',NULL,'2026-08-05')")).all() == []
            conn.execute(text("DROP VIEW mlb.bet_ledger_agg_v"))
            conn.execute(text("ALTER VIEW mlb.acceptance_original_ledger RENAME TO bet_ledger_agg_v"))
        from backend import db as guarded
        from v2.markets import writer
        monkeypatch.setattr(writer, "engine", db)
        now = pd.Timestamp.now(tz="UTC")
        day = now.date()
        with db.begin() as conn:
            conn.execute(text("""INSERT INTO games
              (game_pk, game_date, home_team, away_team, start_time)
              VALUES (1, :day, 'BOS', 'NYY', clock_timestamp() + interval '1 hour')"""),
                         {"day": day})
        rows = [dict(game_pk=1, date=day, team=team, expected_runs=4.0,
                     prediction_updated_at=now - pd.Timedelta(minutes=1))
                for team in ("BOS", "NYY")]
        def fail_archive(conn, cursor, statement, parameters, context, executemany):
            if statement.startswith("INSERT INTO model_outputs_season"):
                raise RuntimeError("archive unavailable")
        event.listen(db, "before_cursor_execute", fail_archive)
        with pytest.raises(RuntimeError, match="archive unavailable"):
            writer.publish_forecasts(rows)
        event.remove(db, "before_cursor_execute", fail_archive)
        with db.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM model_outputs")).scalar_one() == 0
        writer.publish_forecasts(rows)
        writer.publish_forecasts([{**row, "expected_runs": 99,
                                  "prediction_updated_at": now - pd.Timedelta(minutes=2)} for row in rows])
        with db.begin() as conn:
            for table in ("model_outputs", "model_outputs_season"):
                assert conn.execute(text(f"SELECT expected_runs FROM {table}")).scalars().all() == [4, 4]
            conn.execute(text("UPDATE games SET start_time = clock_timestamp() - interval '1 second'"))
        writer.publish_forecasts([{**row, "expected_runs": 99,
                                  "prediction_updated_at": now} for row in rows])
        with db.connect() as conn:
            assert conn.execute(text("SELECT expected_runs FROM model_outputs")).scalars().all() == [4, 4]
        import pipeline
        monkeypatch.setattr(pipeline, "engine", db)
        moved_day = day + pd.Timedelta(days=1)
        with db.begin() as conn:
            pipeline._batch_upsert_games(conn, pd.DataFrame([dict(
                game_pk=1, game_date=moved_day, game_type="D", home_team="BOS",
                away_team="NYY", start_time=now + pd.Timedelta(days=1),
                home_score=None, away_score=None, status="Scheduled", venue="Fenway Park",
            )]))
            moved = conn.execute(text("SELECT game_date, game_type, start_time FROM games WHERE game_pk=1")).one()
            assert moved.game_date == moved_day and moved.game_type == "D"
            assert moved.start_time == now + pd.Timedelta(days=1)
        starter = pd.DataFrame([dict(game_pk=1, is_home=True, game_date=day, team="BOS", pitcher_name="Original", pitcher_id=10)])
        pipeline.upsert_probable_starters(starter)
        pipeline.upsert_probable_starters(starter.assign(pitcher_name=None, pitcher_id=None))
        with db.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM probable_starters")).scalar_one() == 0
        pipeline.upsert_probable_starters(starter.assign(pitcher_name="Replacement", pitcher_id=11))
        with db.connect() as conn:
            assert conn.execute(text("SELECT pitcher_id FROM probable_starters")).scalar_one() == 11
        read_only = create_engine(local_url)
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.delenv("MLBMODEL_DB_WRITES", raising=False)
        event.listen(read_only, "connect", guarded._configure_read_only_session)
        for sql in ("/* comment */ DELETE FROM mlb.games",
                    "WITH removed AS (DELETE FROM mlb.games RETURNING *) SELECT * FROM removed",
                    "SELECT 1; DELETE FROM mlb.games"):
            with pytest.raises(DBAPIError), read_only.begin() as conn:
                conn.execute(text(sql))
        read_only.dispose()
        values = [dict(date=str(day), eval_window=w, total_predictions=2)
                  for w in ("day", "7d", "30d", "season")]
        with db.begin() as conn:
            started = conn.execute(text("SELECT mlb.live_evaluation_started_at()")).scalar_one()
            statement = text("SELECT mlb.publish_live_evaluation(:started, CAST(:rows AS jsonb))")
            conn.execute(statement, {"started": started, "rows": json.dumps(values)})
            stale = [{**row, "total_predictions": 1} for row in values]
            conn.execute(statement, {"started": started - pd.Timedelta(seconds=1), "rows": json.dumps(stale)})
            assert conn.execute(text("SELECT total_predictions FROM model_evaluation")).scalars().all() == [2]*4
            conn.execute(text("UPDATE model_evaluation SET evaluation_state = 'canonical', total_predictions = 3"))
            conn.execute(statement, {"started": started, "rows": json.dumps(values)})
            assert conn.execute(text("SELECT total_predictions FROM model_evaluation")).scalars().all() == [3]*4
        # Completion is separate from Final status and atomic with all windows.
        completion_day = day + pd.Timedelta(days=2)
        with db.begin() as conn:
            conn.execute(text("""INSERT INTO games
                (game_pk, game_date, home_team, away_team, home_score, away_score, status)
                VALUES (2, :day, 'BOS', 'NYY', 4, 2, 'Final')"""), {"day": completion_day})
            assert conn.execute(text("SELECT input_version FROM mlb.live_evaluation_status(2)")).scalar_one() is None
            conn.execute(text("""INSERT INTO model_outputs_season
                (game_pk, date, team, expected_runs, win_prob)
                VALUES (2, :day, 'BOS', 4, 0.6)"""), {"day": completion_day})
            version, completed = conn.execute(text("SELECT * FROM mlb.live_evaluation_status(2)")).one()
            assert version and completed is None
            completed_rows = [{**r, "date": str(completion_day)} for r in values]
            complete = text("""SELECT mlb.complete_live_evaluation(
                2, :version, clock_timestamp(), CAST(:rows AS jsonb))""")
            args = {"version": version, "rows": json.dumps(completed_rows)}
            conn.execute(text("""CREATE FUNCTION mlb.fail_completion() RETURNS trigger
                LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'completion unavailable'; END $$"""))
            conn.execute(text("""CREATE TRIGGER fail_completion BEFORE INSERT
                ON mlb.live_evaluation_completions FOR EACH ROW EXECUTE FUNCTION mlb.fail_completion()"""))
            with pytest.raises(DBAPIError, match="completion unavailable"), conn.begin_nested():
                conn.execute(complete, args)
            assert conn.execute(text("SELECT count(*) FROM model_evaluation WHERE date=:day"), {"day": completion_day}).scalar_one() == 0
            assert conn.execute(text("SELECT count(*) FROM mlb.live_evaluation_completions")).scalar_one() == 0
            conn.execute(text("DROP TRIGGER fail_completion ON mlb.live_evaluation_completions"))
            conn.execute(text("DROP FUNCTION mlb.fail_completion()"))
            conn.execute(complete, args)
            assert conn.execute(text("SELECT eval_date FROM mlb.live_evaluation_status(2)")).scalar_one() == completion_day
            duplicate_args = {**args, "rows": json.dumps([{**r, "total_predictions": 99} for r in completed_rows])}
            conn.execute(complete, duplicate_args)
            assert conn.execute(text("SELECT total_predictions FROM model_evaluation WHERE date=:day"), {"day": completion_day}).scalars().all() == [2]*4
            conn.execute(text("UPDATE games SET updated_at=clock_timestamp() WHERE game_pk=2"))
            assert conn.execute(text("SELECT eval_date FROM mlb.live_evaluation_status(2)")).scalar_one() == completion_day
            conn.execute(text("UPDATE games SET home_score=5 WHERE game_pk=2"))
            assert conn.execute(text("SELECT eval_date FROM mlb.live_evaluation_status(2)")).scalar_one() is None
            with pytest.raises(DBAPIError, match="inputs changed"), conn.begin_nested():
                conn.execute(complete, args)
            args["version"] = conn.execute(text("SELECT input_version FROM mlb.live_evaluation_status(2)")).scalar_one()
            conn.execute(complete, args)
            conn.execute(text("UPDATE model_outputs_season SET win_prob=0.7 WHERE game_pk=2"))
            assert conn.execute(text("SELECT eval_date FROM mlb.live_evaluation_status(2)")).scalar_one() is None
            for role in ("anon", "authenticated"):
                assert not conn.execute(text("SELECT has_function_privilege(:role, 'mlb.complete_live_evaluation(integer,text,timestamptz,jsonb)', 'EXECUTE')"), {"role": role}).scalar_one()
                assert not conn.execute(text("SELECT has_table_privilege(:role, 'mlb.live_evaluation_completions', 'SELECT')"), {"role": role}).scalar_one()
            # Keep the canonical fixture below independent of these extra games.
            conn.execute(text("DELETE FROM model_outputs_season WHERE game_pk=2"))
            conn.execute(text("DELETE FROM games WHERE game_pk=2"))
            conn.execute(text("DELETE FROM model_evaluation WHERE date=:day"), {"day": completion_day})
        from backend import evaluate_model
        from backend.metrics import financial_summary, probabilistic_summary
        monkeypatch.setattr(evaluate_model, "engine", db)
        evaluate_model._write_evaluation_row(day, "day", {"mae": 1.0, "ml_predictions": 1, "ml_accuracy": 1.0},
                                             {"roi": 50.0, "net_profit_units": 2.0})
        evaluate_model._write_evaluation_row(day, "day", {"mae": 1.0, "ml_predictions": 0, "ml_accuracy": None}, financial_summary(pd.DataFrame()))
        with db.connect() as conn:
            assert conn.execute(text("SELECT net_profit_units FROM model_evaluation WHERE eval_window='day'")).scalar_one() == 0
            assert conn.execute(text("SELECT roi FROM model_evaluation WHERE eval_window='day'")).scalar_one() is None
        with db.connect() as conn:
            assert conn.execute(text("SELECT ml_accuracy FROM model_evaluation WHERE eval_window='day'")).scalar_one() is None
        evaluate_model._write_calibration(day, [{"bin_mid": .5, "predicted_mean": .5, "observed_rate": .5, "count": 10}])
        evaluate_model._write_calibration(day, [])
        with db.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM model_calibration")).scalar_one() == 0
        fixture = json.loads((root / "tests/fixtures/contracts/v1/mlb-evaluation.json").read_text())
        with db.begin() as conn:
            for prediction, game in zip(fixture["predictions"], fixture["games"], strict=True):
                conn.execute(text("""INSERT INTO games
                    (game_pk, game_date, home_team, away_team, home_score, away_score, status)
                    VALUES (:pk, :date, 'H', 'A', 4, 2, 'Final')"""),
                    {"pk": game["game_pk"] + 100, "date": game["game_date"]})
                conn.execute(text("""INSERT INTO model_outputs_season
                    (game_pk, date, team, expected_runs, win_prob, ev_flag, run_line_ev_flag, total_play)
                    VALUES (:pk, :date, 'H', 4, 0.6, 'No Play', 'No Play', 'No Play')"""),
                    {"pk": prediction["game_pk"] + 100, "date": prediction["date"]})
        evaluate_model.main(as_of=(pd.Timestamp(fixture["date"]) + pd.Timedelta(days=1)).date())
        with db.connect() as conn:
            counts = dict(conn.execute(text("""SELECT eval_window, total_predictions
                FROM model_evaluation WHERE date = :date"""), {"date": fixture["date"]}).all())
            assert counts == fixture["window_counts"]
        narrow, wide = [0.0]*21, [0.0]*21
        narrow[4], wide[0], wide[8] = 1.0, .5, .5
        first = probabilistic_summary([.5], [1], histograms=[narrow], actual_runs=[6])
        second = probabilistic_summary([.5], [1], histograms=[wide], actual_runs=[6])
        assert first["interval_coverage_80"] == 0 and second["interval_coverage_80"] == 1
        assert second["interval_coverage_predictions"] == 1
    finally:
        db.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE "{name}"'))
        admin.dispose()


@pytest.mark.database
def test_betting_headline_preserves_canonical_ledger():
    value = os.getenv("MLB_TEST_DATABASE_URL")
    if not value:
        pytest.skip("requires a disposable local PostgreSQL admin database")
    url = make_url(value)
    assert url.host in ("127.0.0.1", "localhost")
    name = "mlb_headline_" + uuid.uuid4().hex
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    local_url = url.set(database=name)
    db = create_engine(local_url)
    try:
        with db.begin() as conn:
            conn.execute(text("""CREATE FUNCTION public.site_revalidate() RETURNS trigger
                LANGUAGE plpgsql AS $$ BEGIN RETURN NULL; END $$"""))
        root = Path(__file__).parents[1]
        for path in (root / "backend/sql/baseline/2026_10_05.sql",
                     root / "backend/sql/betting_headline.sql"):
            subprocess.run(["psql", "-X", local_url.render_as_string(hide_password=False),
                            "-v", "ON_ERROR_STOP=1", "-f", str(path)],
                           check=True, capture_output=True)
        with db.begin() as conn:
            assert conn.execute(text("SELECT * FROM mlb.betting_headline()")).all() == []
            # Both archive eras, duplicate source rows at cutover, ties, pending,
            # cancelled (void), missing scores and a rescheduled date mismatch.
            conn.execute(text("""INSERT INTO mlb.games
                (game_pk,game_date,home_team,away_team,home_score,away_score,status)
                SELECT n, CASE WHEN n=1 THEN '2026-05-11'::date ELSE '2026-05-12'::date END,
                  'BOS','NYY',CASE WHEN n=6 THEN NULL ELSE 4 END,
                  CASE WHEN n=3 THEN 4 ELSE 2 END,
                  CASE n WHEN 4 THEN 'Scheduled' WHEN 5 THEN 'Cancelled' ELSE 'Final' END
                FROM generate_series(1,9) n"""))
            for table in ("model_outputs_season", "model_outputs_season_v1_archive"):
                conn.execute(text(f"""INSERT INTO mlb.{table}
                    (game_pk,date,team,ev_flag,run_line_ev_flag,total_play,
                     win_prob,p_cover,p_over,p_under,moneyline,spread,spread_odds,
                     total,total_over_odds,total_under_odds,
                     kelly_quarter_ml,kelly_quarter_rl,kelly_quarter_total)
                    SELECT game_pk, game_date + CASE WHEN game_pk=7 THEN 1 ELSE 0 END,
                      team, CASE WHEN game_pk=8 THEN 'No Play' ELSE team END,
                      CASE WHEN game_pk=8 THEN 'No Play' ELSE team END,
                      CASE WHEN game_pk=8 THEN 'No Play' ELSE 'Over' END,
                      CASE WHEN game_pk=9 THEN .51 ELSE .75 END,
                      CASE WHEN game_pk=9 THEN .51 ELSE .75 END,
                      CASE WHEN game_pk=9 THEN .51 ELSE .75 END, .75,
                      100,-1.5,100,6,100,100,.1,.2,.3
                    FROM mlb.games CROSS JOIN (VALUES ('BOS'),('NYY')) t(team)"""))
            # Current outputs never replace the frozen season population.
            conn.execute(text("""INSERT INTO mlb.model_outputs (game_pk,date,team,win_prob)
                VALUES (2,'2026-05-12','BOS',.01)"""))
            def compare():
                ledger = conn.execute(text("""SELECT * FROM mlb.bet_ledger_v
                    ORDER BY date,game_pk,bet_type,team""")).mappings().all()
                expected = {}
                for row in ledger:
                    item = expected.setdefault(row['bet_type'], dict(
                        bet_type=row['bet_type'], wins=0, losses=0, pushes=0,
                        total_stake=0., total_payout=0.))
                    item['wins'] += int(row['won'])
                    item['pushes'] += int(row['push'])
                    item['losses'] += 1-int(row['won'])-int(row['push'])
                    item['total_stake'] += float(row['stake'])
                    item['total_payout'] += float(row['payout'])
                actual = conn.execute(text("SELECT * FROM mlb.betting_headline()")).mappings().all()
                assert {r['bet_type']: dict(r) for r in actual} == expected
                return ledger, expected
            ledger, before = compare()
            assert {r['game_pk'] for r in ledger} == {1,2,3}
            assert len(ledger) == 15  # Two sides of ML/RL, one total per game.
            assert before['total']['pushes'] == 2
            assert before['ml']['losses'] == 4  # Existing tie semantics preserved.
            frozen = conn.execute(text("SELECT row_to_json(t) FROM mlb.model_outputs_season t ORDER BY game_pk,team")).scalars().all()
            conn.execute(text("UPDATE mlb.games SET home_score=1 WHERE game_pk=2"))
            _, corrected = compare()
            assert corrected != before
            assert corrected['total']['pushes'] == 1
            assert conn.execute(text("SELECT row_to_json(t) FROM mlb.model_outputs_season t ORDER BY game_pk,team")).scalars().all() == frozen
            conn.execute(text("UPDATE mlb.model_outputs_season SET run_line_ev_flag='No Play', total_play='No Play'"))
            conn.execute(text("UPDATE mlb.model_outputs_season_v1_archive SET run_line_ev_flag='No Play', total_play='No Play'"))
            _, selected = compare()
            assert set(selected) == {'ml'}
            assert not conn.execute(text("SELECT prosecdef FROM pg_proc WHERE oid='mlb.betting_headline()'::regprocedure")).scalar_one()
            for role in ('anon', 'authenticated', 'service_role'):
                conn.execute(text(f'SET LOCAL ROLE {role}'))
                ledger, visible = compare()
                if role in ('anon', 'authenticated'):
                    assert set(visible) == {'ml'}
                # A local service_role may lack Supabase's BYPASSRLS attribute.
                # The invoker function must always match that caller's ledger.
                conn.execute(text('RESET ROLE'))
    finally:
        db.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE "{name}"'))
        admin.dispose()
