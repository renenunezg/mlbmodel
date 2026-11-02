BEGIN;

-- Award forecasts, frozen when published. One row per candidate shown: the top of each ballot race (MVP, CY, ROY,
-- RP) and the three finalists of each Gold Glove race (GG, with position set). rank is null only for a winner who
-- was outside the published board, added by the grader. win_probability is null where the model has not earned a
-- probability (Gold Glove pitcher and utility), vote_share where no votes exist (RP, GG). won is null until graded.
CREATE TABLE mlb.award_boards (
  season smallint NOT NULL,
  league text NOT NULL CHECK (league IN ('AL', 'NL')),
  award text NOT NULL CHECK (award IN ('MVP', 'CY', 'ROY', 'RP', 'GG')),
  position text NOT NULL DEFAULT '',
  player_id integer NOT NULL,
  player_name text NOT NULL,
  team text,
  rank smallint,
  win_probability real,
  vote_share real,
  stat_line text NOT NULL DEFAULT '',
  won boolean,
  PRIMARY KEY (season, league, award, position, player_id)
);
-- One row per fitted model: when it froze, and its expanding-window backtest.
CREATE TABLE mlb.award_models (
  season smallint NOT NULL,
  model text NOT NULL,
  as_of timestamptz NOT NULL,
  model_version text NOT NULL,
  backtest jsonb NOT NULL,
  PRIMARY KEY (season, model)
);
ALTER TABLE mlb.award_boards ENABLE ROW LEVEL SECURITY;
ALTER TABLE mlb.award_models ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.award_boards FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY public_read ON mlb.award_models FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.award_boards, mlb.award_models TO anon, authenticated;
GRANT SELECT, INSERT ON mlb.award_models TO service_role;
GRANT SELECT, INSERT, UPDATE ON mlb.award_boards TO service_role;

-- A forecast cannot move after it is published. The grader may only set won, once, on a row that has none.
CREATE FUNCTION mlb.refuse_award_board_change() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'award_boards is frozen (DELETE)';
  END IF;
  IF OLD.won IS NOT NULL OR (to_jsonb(NEW) - 'won') <> (to_jsonb(OLD) - 'won') THEN
    RAISE EXCEPTION 'award_boards is frozen: only an ungraded row can take won';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER refuse_award_board_change BEFORE UPDATE OR DELETE ON mlb.award_boards
  FOR EACH ROW EXECUTE FUNCTION mlb.refuse_award_board_change();
CREATE TRIGGER refuse_truncate BEFORE TRUNCATE ON mlb.award_boards
  FOR EACH STATEMENT EXECUTE FUNCTION mlb.refuse_truncate();

CREATE FUNCTION mlb.refuse_award_model_change() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  RAISE EXCEPTION 'award_models is write-once (%)', TG_OP;
END $$;
CREATE TRIGGER refuse_award_model_change BEFORE UPDATE OR DELETE ON mlb.award_models
  FOR EACH ROW EXECUTE FUNCTION mlb.refuse_award_model_change();
CREATE TRIGGER refuse_truncate BEFORE TRUNCATE ON mlb.award_models
  FOR EACH STATEMENT EXECUTE FUNCTION mlb.refuse_truncate();

CREATE TRIGGER site_revalidate AFTER INSERT OR UPDATE OR DELETE ON mlb.award_boards
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
CREATE TRIGGER site_revalidate AFTER INSERT OR UPDATE OR DELETE ON mlb.award_models
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

NOTIFY pgrst, 'reload schema';

COMMIT;
