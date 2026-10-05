--
-- PostgreSQL database dump
--

\restrict SL8fUNpNYBGbzkVY1pJrGMqsqw6e6jedy8EKkEHkgv7vMfcjFLemdcmzHFji6SY

-- Dumped from database version 17.6
-- Dumped by pg_dump version 17.9 (Postgres.app)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: mlb; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA mlb;


--
-- Name: bet_record_summary(text, text); Type: FUNCTION; Schema: mlb; Owner: -
--

CREATE FUNCTION mlb.bet_record_summary(p_from text DEFAULT NULL::text, p_team text DEFAULT NULL::text) RETURNS TABLE(bet_type text, wins bigint, losses bigint, pushes bigint)
    LANGUAGE sql STABLE
    SET search_path TO 'mlb'
    AS $$
  select bet_type,
         count(*) filter (where won) as wins,
         count(*) filter (where not won and not push) as losses,
         count(*) filter (where push) as pushes
  from mlb.bet_ledger_agg_v
  where (nullif(p_from, '') is null or (date)::date >= (p_from)::date)
    and (nullif(p_team, '') is null or team = p_team)
  group by bet_type;
$$;


--
-- Name: set_updated_at_model_outputs_season(); Type: FUNCTION; Schema: mlb; Owner: -
--

CREATE FUNCTION mlb.set_updated_at_model_outputs_season() RETURNS trigger
    LANGUAGE plpgsql
    SET search_path TO 'public', 'pg_temp'
    AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: games; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.games (
    game_pk integer NOT NULL,
    game_date date NOT NULL,
    home_team character varying(3) NOT NULL,
    away_team character varying(3) NOT NULL,
    home_score integer,
    away_score integer,
    status character varying(20) DEFAULT 'Scheduled'::character varying NOT NULL,
    venue character varying(100),
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    start_time timestamp with time zone,
    game_type text,
    CONSTRAINT games_game_type_check CHECK ((game_type = ANY (ARRAY['R'::text, 'F'::text, 'D'::text, 'L'::text, 'W'::text])))
);


--
-- Name: model_outputs_season; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.model_outputs_season (
    game_pk bigint,
    date timestamp without time zone,
    team text,
    starter text,
    expected_runs double precision,
    win_prob double precision,
    our_odds bigint,
    expected_runs_p10 double precision,
    expected_runs_p50 double precision,
    expected_runs_p90 double precision,
    total_p10 double precision,
    total_p50 double precision,
    total_p90 double precision,
    win_prob_p10 double precision,
    win_prob_p90 double precision,
    moneyline double precision,
    total double precision,
    spread double precision,
    spread_odds double precision,
    our_total double precision,
    total_diff double precision,
    total_play text,
    ev_flag text,
    run_line_ev_flag text,
    ml_confidence double precision,
    run_line_confidence double precision,
    high_variance_flag text,
    kelly_full_ml numeric,
    kelly_quarter_ml numeric,
    kelly_full_rl numeric,
    kelly_quarter_rl numeric,
    kelly_full_total numeric,
    kelly_quarter_total numeric,
    p_cover numeric,
    p_over numeric,
    p_under numeric,
    total_over_odds numeric,
    total_under_odds numeric,
    lineups_locked boolean DEFAULT false,
    lineup_source text,
    prediction_updated_at timestamp with time zone DEFAULT now(),
    posterior_age_days integer,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    lineup_hash text,
    start_time timestamp with time zone,
    runs_hist jsonb,
    prediction_context jsonb
);


--
-- Name: COLUMN model_outputs_season.runs_hist; Type: COMMENT; Schema: mlb; Owner: -
--

COMMENT ON COLUMN mlb.model_outputs_season.runs_hist IS 'Empirical PMF of simulated runs scored by this team, 21 bins (0..20 runs).';


--
-- Name: model_outputs_season_v1_archive; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.model_outputs_season_v1_archive (
    game_pk bigint,
    date timestamp without time zone,
    team text,
    starter text,
    expected_runs double precision,
    win_prob double precision,
    our_odds bigint,
    moneyline double precision,
    total double precision,
    spread double precision,
    spread_odds double precision,
    our_total double precision,
    total_diff double precision,
    total_play text,
    ev_flag text,
    run_line_ev_flag text,
    ml_confidence double precision,
    run_line_confidence double precision,
    high_variance_flag text,
    kelly_full_ml numeric,
    kelly_quarter_ml numeric,
    kelly_full_rl numeric,
    kelly_quarter_rl numeric,
    kelly_full_total numeric,
    kelly_quarter_total numeric,
    p_cover numeric,
    p_over numeric,
    p_under numeric,
    total_over_odds numeric,
    total_under_odds numeric,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: bet_ledger_v; Type: VIEW; Schema: mlb; Owner: -
--

CREATE VIEW mlb.bet_ledger_v WITH (security_invoker='true') AS
 WITH preds AS (
         SELECT model_outputs_season_v1_archive.game_pk,
            model_outputs_season_v1_archive.date,
            model_outputs_season_v1_archive.team,
            model_outputs_season_v1_archive.ev_flag,
            model_outputs_season_v1_archive.run_line_ev_flag,
            model_outputs_season_v1_archive.total_play,
            model_outputs_season_v1_archive.win_prob,
            model_outputs_season_v1_archive.p_cover,
            model_outputs_season_v1_archive.p_over,
            model_outputs_season_v1_archive.p_under,
            model_outputs_season_v1_archive.moneyline,
            model_outputs_season_v1_archive.spread,
            model_outputs_season_v1_archive.spread_odds,
            model_outputs_season_v1_archive.total,
            model_outputs_season_v1_archive.total_over_odds,
            model_outputs_season_v1_archive.total_under_odds,
            model_outputs_season_v1_archive.kelly_quarter_ml,
            model_outputs_season_v1_archive.kelly_quarter_rl,
            model_outputs_season_v1_archive.kelly_quarter_total
           FROM mlb.model_outputs_season_v1_archive
          WHERE ((model_outputs_season_v1_archive.date)::date < '2026-05-12'::date)
        UNION ALL
         SELECT model_outputs_season.game_pk,
            model_outputs_season.date,
            model_outputs_season.team,
            model_outputs_season.ev_flag,
            model_outputs_season.run_line_ev_flag,
            model_outputs_season.total_play,
            model_outputs_season.win_prob,
            model_outputs_season.p_cover,
            model_outputs_season.p_over,
            model_outputs_season.p_under,
            model_outputs_season.moneyline,
            model_outputs_season.spread,
            model_outputs_season.spread_odds,
            model_outputs_season.total,
            model_outputs_season.total_over_odds,
            model_outputs_season.total_under_odds,
            model_outputs_season.kelly_quarter_ml,
            model_outputs_season.kelly_quarter_rl,
            model_outputs_season.kelly_quarter_total
           FROM mlb.model_outputs_season
          WHERE ((model_outputs_season.date)::date >= '2026-05-12'::date)
        ), joined AS (
         SELECT m.game_pk,
            m.date,
            m.team,
            m.ev_flag,
            m.run_line_ev_flag,
            m.total_play,
            m.win_prob,
            m.p_cover,
            m.p_over,
            m.p_under,
            m.moneyline,
            m.spread,
            m.spread_odds,
            m.total,
            m.total_over_odds,
            m.total_under_odds,
            m.kelly_quarter_ml,
            m.kelly_quarter_rl,
            m.kelly_quarter_total,
            g.home_team,
            g.home_score,
            g.away_score,
            (g.home_score + g.away_score) AS game_total,
                CASE
                    WHEN ((g.home_team)::text = m.team) THEN g.home_score
                    ELSE g.away_score
                END AS team_score,
                CASE
                    WHEN ((g.home_team)::text = m.team) THEN g.away_score
                    ELSE g.home_score
                END AS opp_score,
                CASE
                    WHEN ((g.home_team)::text = m.team) THEN (g.home_score - g.away_score)
                    ELSE (g.away_score - g.home_score)
                END AS margin
           FROM (preds m
             JOIN mlb.games g ON (((g.game_pk = m.game_pk) AND (g.game_date = (m.date)::date))))
          WHERE (((g.status)::text = 'Final'::text) AND (g.home_score IS NOT NULL) AND (g.away_score IS NOT NULL))
        ), ml AS (
         SELECT joined.date,
            joined.team,
            joined.game_pk,
            'ml'::text AS bet_type,
            joined.kelly_quarter_ml AS stake,
                CASE
                    WHEN (joined.moneyline > (0)::double precision) THEN ((1)::double precision + (joined.moneyline / (100.0)::double precision))
                    ELSE ((1)::double precision + ((100.0)::double precision / abs(joined.moneyline)))
                END AS decimal_odds,
            (joined.moneyline)::numeric AS american_odds,
            NULL::text AS totals_side,
            (joined.team_score > joined.opp_score) AS won,
            (joined.win_prob -
                CASE
                    WHEN (joined.moneyline > (0)::double precision) THEN ((100.0)::double precision / (joined.moneyline + (100.0)::double precision))
                    ELSE ((- joined.moneyline) / ((- joined.moneyline) + (100.0)::double precision))
                END) AS edge,
            false AS push
           FROM joined
          WHERE ((joined.ev_flag = joined.team) AND (joined.moneyline IS NOT NULL) AND (joined.kelly_quarter_ml IS NOT NULL) AND (joined.kelly_quarter_ml > (0)::numeric) AND (joined.win_prob IS NOT NULL))
        ), rl AS (
         SELECT joined.date,
            joined.team,
            joined.game_pk,
            'rl'::text AS bet_type,
            joined.kelly_quarter_rl AS stake,
                CASE
                    WHEN (joined.spread_odds > (0)::double precision) THEN ((1)::double precision + (joined.spread_odds / (100.0)::double precision))
                    ELSE ((1)::double precision + ((100.0)::double precision / abs(joined.spread_odds)))
                END AS decimal_odds,
            NULL::numeric AS american_odds,
            NULL::text AS totals_side,
                CASE
                    WHEN (joined.spread < (0)::double precision) THEN ((joined.margin)::double precision >= abs(joined.spread))
                    WHEN (joined.spread > (0)::double precision) THEN ((joined.team_score > joined.opp_score) OR ((joined.margin)::double precision >= (- joined.spread)))
                    ELSE false
                END AS won,
            ((joined.p_cover)::double precision -
                CASE
                    WHEN (joined.spread_odds > (0)::double precision) THEN ((100.0)::double precision / (joined.spread_odds + (100.0)::double precision))
                    ELSE ((- joined.spread_odds) / ((- joined.spread_odds) + (100.0)::double precision))
                END) AS edge,
            false AS push
           FROM joined
          WHERE ((joined.run_line_ev_flag = joined.team) AND (joined.spread IS NOT NULL) AND (joined.spread_odds IS NOT NULL) AND (joined.kelly_quarter_rl IS NOT NULL) AND (joined.kelly_quarter_rl > (0)::numeric) AND (joined.p_cover IS NOT NULL))
        ), tot_raw AS (
         SELECT joined.date,
            joined.team,
            joined.game_pk,
            'total'::text AS bet_type,
            joined.kelly_quarter_total AS stake,
            lower(joined.total_play) AS totals_side,
                CASE
                    WHEN (joined.total_play = 'Over'::text) THEN
                    CASE
                        WHEN (joined.total_over_odds > (0)::numeric) THEN ((1)::numeric + (joined.total_over_odds / 100.0))
                        ELSE ((1)::numeric + (100.0 / abs(joined.total_over_odds)))
                    END
                    ELSE
                    CASE
                        WHEN (joined.total_under_odds > (0)::numeric) THEN ((1)::numeric + (joined.total_under_odds / 100.0))
                        ELSE ((1)::numeric + (100.0 / abs(joined.total_under_odds)))
                    END
                END AS decimal_odds,
            NULL::numeric AS american_odds,
                CASE
                    WHEN (joined.total_play = 'Over'::text) THEN ((joined.game_total)::double precision > joined.total)
                    WHEN (joined.total_play = 'Under'::text) THEN ((joined.game_total)::double precision < joined.total)
                    ELSE false
                END AS won,
            (
                CASE
                    WHEN (joined.total_play = 'Over'::text) THEN joined.p_over
                    ELSE joined.p_under
                END -
                CASE
                    WHEN (joined.total_play = 'Over'::text) THEN
                    CASE
                        WHEN (joined.total_over_odds > (0)::numeric) THEN (100.0 / (joined.total_over_odds + 100.0))
                        ELSE ((- joined.total_over_odds) / ((- joined.total_over_odds) + 100.0))
                    END
                    ELSE
                    CASE
                        WHEN (joined.total_under_odds > (0)::numeric) THEN (100.0 / (joined.total_under_odds + 100.0))
                        ELSE ((- joined.total_under_odds) / ((- joined.total_under_odds) + 100.0))
                    END
                END) AS edge,
            ((joined.game_total)::double precision = joined.total) AS push,
            row_number() OVER (PARTITION BY joined.game_pk ORDER BY joined.team) AS rn
           FROM joined
          WHERE ((joined.total_play = ANY (ARRAY['Over'::text, 'Under'::text])) AND (joined.total IS NOT NULL) AND (joined.game_total IS NOT NULL) AND (joined.kelly_quarter_total IS NOT NULL) AND (joined.kelly_quarter_total > (0)::numeric) AND (
                CASE
                    WHEN (joined.total_play = 'Over'::text) THEN joined.p_over
                    ELSE joined.p_under
                END IS NOT NULL) AND (
                CASE
                    WHEN (joined.total_play = 'Over'::text) THEN joined.total_over_odds
                    ELSE joined.total_under_odds
                END IS NOT NULL))
        ), tot AS (
         SELECT tot_raw.date,
            tot_raw.team,
            tot_raw.game_pk,
            tot_raw.bet_type,
            tot_raw.stake,
            tot_raw.decimal_odds,
            tot_raw.american_odds,
            tot_raw.totals_side,
            tot_raw.won,
            tot_raw.edge,
            tot_raw.push
           FROM tot_raw
          WHERE (tot_raw.rn = 1)
        )
 SELECT date,
    team,
    game_pk,
    bet_type,
    stake,
    decimal_odds,
    american_odds,
    totals_side,
    won,
    edge,
    ((stake)::double precision *
        CASE
            WHEN won THEN decimal_odds
            WHEN push THEN (1)::double precision
            ELSE (0)::double precision
        END) AS payout,
    push
   FROM ( SELECT ml.date,
            ml.team,
            ml.game_pk,
            ml.bet_type,
            ml.stake,
            ml.decimal_odds,
            ml.american_odds,
            ml.totals_side,
            ml.won,
            ml.edge,
            ml.push
           FROM ml
          WHERE (ml.edge >= (0.045)::double precision)
        UNION ALL
         SELECT rl.date,
            rl.team,
            rl.game_pk,
            rl.bet_type,
            rl.stake,
            rl.decimal_odds,
            rl.american_odds,
            rl.totals_side,
            rl.won,
            rl.edge,
            rl.push
           FROM rl
          WHERE (rl.edge >= (0.045)::double precision)
        UNION ALL
         SELECT tot.date,
            tot.team,
            tot.game_pk,
            tot.bet_type,
            tot.stake,
            tot.decimal_odds,
            tot.american_odds,
            tot.totals_side,
            tot.won,
            tot.edge,
            tot.push
           FROM tot
          WHERE (tot.edge >= 0.065)) u;


--
-- Name: bet_ledger_agg_v; Type: VIEW; Schema: mlb; Owner: -
--

CREATE VIEW mlb.bet_ledger_agg_v WITH (security_invoker='true') AS
 SELECT date,
    team,
    game_pk,
    bet_type,
    totals_side,
        CASE
            WHEN (american_odds < (0)::numeric) THEN 'fav'::text
            WHEN (american_odds > (0)::numeric) THEN 'dog'::text
            ELSE NULL::text
        END AS ml_side,
    stake,
    payout,
    won,
    edge,
    american_odds,
    decimal_odds,
    push
   FROM mlb.bet_ledger_v;


--
-- Name: bullpen_daily; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.bullpen_daily (
    game_date date NOT NULL,
    team character varying(3) NOT NULL,
    reliever_outs integer DEFAULT 0 NOT NULL,
    starter_outs integer DEFAULT 0 NOT NULL,
    n_relievers integer DEFAULT 0 NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: bullpen_stats; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.bullpen_stats (
    id integer NOT NULL,
    team character varying(3) NOT NULL,
    season integer DEFAULT 2026 NOT NULL,
    ip numeric(6,1),
    era numeric(5,2),
    fip numeric(5,2),
    xfip numeric(5,2),
    siera numeric(5,2),
    whip numeric(5,3),
    k_9 numeric(5,2),
    bb_9 numeric(5,2),
    hr_9 numeric(5,2),
    rhp_ip_share numeric
);


--
-- Name: bullpen_stats_id_seq; Type: SEQUENCE; Schema: mlb; Owner: -
--

CREATE SEQUENCE mlb.bullpen_stats_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: bullpen_stats_id_seq; Type: SEQUENCE OWNED BY; Schema: mlb; Owner: -
--

ALTER SEQUENCE mlb.bullpen_stats_id_seq OWNED BY mlb.bullpen_stats.id;


--
-- Name: experiment_runs; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.experiment_runs (
    id integer NOT NULL,
    run_date timestamp with time zone DEFAULT now(),
    git_sha text,
    hyperparameters jsonb,
    best_cv_mae numeric,
    feature_list text[],
    notes text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: experiment_runs_id_seq; Type: SEQUENCE; Schema: mlb; Owner: -
--

CREATE SEQUENCE mlb.experiment_runs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: experiment_runs_id_seq; Type: SEQUENCE OWNED BY; Schema: mlb; Owner: -
--

ALTER SEQUENCE mlb.experiment_runs_id_seq OWNED BY mlb.experiment_runs.id;


--
-- Name: live_win_probability; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.live_win_probability (
    game_pk integer NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    source_timestamp timestamp with time zone NOT NULL,
    payload jsonb NOT NULL,
    CONSTRAINT live_win_probability_check CHECK ((((payload ->> 'schema_version'::text) = '1'::text) AND (((payload ->> 'game_pk'::text))::integer = game_pk))),
    CONSTRAINT live_win_probability_game_pk_check CHECK ((game_pk > 0))
);


--
-- Name: model_calibration; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.model_calibration (
    id integer NOT NULL,
    date date NOT NULL,
    bin_mid numeric NOT NULL,
    predicted_mean numeric,
    observed_rate numeric,
    count integer,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: model_calibration_id_seq; Type: SEQUENCE; Schema: mlb; Owner: -
--

CREATE SEQUENCE mlb.model_calibration_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: model_calibration_id_seq; Type: SEQUENCE OWNED BY; Schema: mlb; Owner: -
--

ALTER SEQUENCE mlb.model_calibration_id_seq OWNED BY mlb.model_calibration.id;


--
-- Name: model_edge_buckets; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.model_edge_buckets (
    id integer NOT NULL,
    date date NOT NULL,
    eval_window character varying NOT NULL,
    bucket_label text NOT NULL,
    n_bets integer,
    hit_rate numeric,
    roi numeric,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: model_edge_buckets_id_seq; Type: SEQUENCE; Schema: mlb; Owner: -
--

CREATE SEQUENCE mlb.model_edge_buckets_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: model_edge_buckets_id_seq; Type: SEQUENCE OWNED BY; Schema: mlb; Owner: -
--

ALTER SEQUENCE mlb.model_edge_buckets_id_seq OWNED BY mlb.model_edge_buckets.id;


--
-- Name: model_evaluation; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.model_evaluation (
    id integer NOT NULL,
    date date NOT NULL,
    total_correct integer,
    total_predictions integer,
    total_accuracy numeric(5,2),
    ml_correct integer,
    ml_predictions integer,
    ml_accuracy numeric(5,2),
    run_line_correct integer,
    run_line_predictions integer,
    run_line_accuracy numeric(5,2),
    average_total_diff numeric(5,2),
    average_win_prob numeric(5,2),
    created_at timestamp with time zone DEFAULT now(),
    eval_window character varying DEFAULT 'day'::character varying,
    mae numeric,
    rmse numeric,
    mape numeric,
    r2 numeric,
    brier_score numeric,
    log_loss numeric,
    sharpness numeric,
    interval_coverage_80 numeric,
    roi numeric,
    sharpe numeric,
    sortino numeric,
    max_drawdown numeric,
    total_staked_units numeric,
    net_profit_units numeric,
    equity_end_units numeric,
    totals_correct integer,
    totals_predictions integer,
    totals_accuracy numeric,
    interval_coverage_50 numeric,
    interval_coverage_90 numeric,
    roi_favorites numeric,
    roi_underdogs numeric,
    n_favorites integer,
    n_underdogs integer,
    avg_ml_line numeric,
    overs_correct integer,
    overs_predictions integer,
    unders_correct integer,
    unders_predictions integer,
    overs_roi numeric,
    unders_roi numeric,
    favorites_correct integer,
    underdogs_correct integer,
    roi_run_line numeric,
    n_run_line integer,
    run_line_bets_correct integer,
    predictions_rewritten boolean DEFAULT false NOT NULL
);


--
-- Name: model_evaluation_id_seq; Type: SEQUENCE; Schema: mlb; Owner: -
--

CREATE SEQUENCE mlb.model_evaluation_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: model_evaluation_id_seq; Type: SEQUENCE OWNED BY; Schema: mlb; Owner: -
--

ALTER SEQUENCE mlb.model_evaluation_id_seq OWNED BY mlb.model_evaluation.id;


--
-- Name: model_feature_importance; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.model_feature_importance (
    id integer NOT NULL,
    date date NOT NULL,
    feature text NOT NULL,
    importance numeric,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: model_feature_importance_id_seq; Type: SEQUENCE; Schema: mlb; Owner: -
--

CREATE SEQUENCE mlb.model_feature_importance_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: model_feature_importance_id_seq; Type: SEQUENCE OWNED BY; Schema: mlb; Owner: -
--

ALTER SEQUENCE mlb.model_feature_importance_id_seq OWNED BY mlb.model_feature_importance.id;


--
-- Name: model_outputs; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.model_outputs (
    game_pk bigint NOT NULL,
    date timestamp without time zone,
    team text NOT NULL,
    starter text,
    expected_runs double precision,
    win_prob double precision,
    our_odds bigint,
    expected_runs_p10 double precision,
    expected_runs_p50 double precision,
    expected_runs_p90 double precision,
    total_p10 double precision,
    total_p50 double precision,
    total_p90 double precision,
    win_prob_p10 double precision,
    win_prob_p90 double precision,
    moneyline double precision,
    total double precision,
    spread double precision,
    spread_odds double precision,
    our_total double precision,
    total_diff double precision,
    total_play text,
    ev_flag text,
    run_line_ev_flag text,
    ml_confidence double precision,
    run_line_confidence double precision,
    high_variance_flag text,
    kelly_full_ml numeric,
    kelly_quarter_ml numeric,
    kelly_full_rl numeric,
    kelly_quarter_rl numeric,
    kelly_full_total numeric,
    kelly_quarter_total numeric,
    p_cover numeric,
    p_over numeric,
    p_under numeric,
    total_over_odds numeric,
    total_under_odds numeric,
    lineups_locked boolean DEFAULT false,
    lineup_source text,
    prediction_updated_at timestamp with time zone DEFAULT now(),
    posterior_age_days integer,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    lineup_hash text,
    start_time timestamp with time zone,
    runs_hist jsonb,
    prediction_context jsonb
);


--
-- Name: COLUMN model_outputs.runs_hist; Type: COMMENT; Schema: mlb; Owner: -
--

COMMENT ON COLUMN mlb.model_outputs.runs_hist IS 'Empirical PMF of simulated runs scored by this team, 21 bins (0..20 runs). Sums to ~1. Written by v2.markets.writer from the raw sim array.';


--
-- Name: model_outputs_season_unified; Type: VIEW; Schema: mlb; Owner: -
--

CREATE VIEW mlb.model_outputs_season_unified WITH (security_invoker='true') AS
 SELECT v1.game_pk,
    v1.date,
    v1.team,
    v1.starter,
    v1.expected_runs,
    v1.win_prob,
    v1.our_odds,
    NULL::double precision AS expected_runs_p10,
    NULL::double precision AS expected_runs_p50,
    NULL::double precision AS expected_runs_p90,
    NULL::double precision AS total_p10,
    NULL::double precision AS total_p50,
    NULL::double precision AS total_p90,
    NULL::double precision AS win_prob_p10,
    NULL::double precision AS win_prob_p90,
    v1.moneyline,
    v1.total,
    v1.spread,
    v1.spread_odds,
    v1.our_total,
    v1.total_diff,
    v1.total_play,
    v1.ev_flag,
    v1.run_line_ev_flag,
    v1.ml_confidence,
    v1.run_line_confidence,
    v1.high_variance_flag,
    v1.kelly_full_ml,
    v1.kelly_quarter_ml,
    v1.kelly_full_rl,
    v1.kelly_quarter_rl,
    v1.kelly_full_total,
    v1.kelly_quarter_total,
    v1.p_cover,
    v1.p_over,
    v1.p_under,
    v1.total_over_odds,
    v1.total_under_odds,
    NULL::boolean AS lineups_locked,
    NULL::text AS lineup_source,
    NULL::timestamp with time zone AS prediction_updated_at,
    NULL::integer AS posterior_age_days,
    v1.created_at,
    v1.updated_at,
    NULL::text AS lineup_hash,
    g.start_time,
    'v1'::text AS model_version,
    g.status AS game_status,
    g.home_team,
    g.away_team,
    g.home_score,
    g.away_score
   FROM (mlb.model_outputs_season_v1_archive v1
     LEFT JOIN mlb.games g USING (game_pk))
  WHERE (((v1.date)::date < '2026-05-12'::date) AND (g.game_date = (v1.date)::date))
UNION ALL
 SELECT m.game_pk,
    m.date,
    m.team,
    m.starter,
    m.expected_runs,
    m.win_prob,
    m.our_odds,
    m.expected_runs_p10,
    m.expected_runs_p50,
    m.expected_runs_p90,
    m.total_p10,
    m.total_p50,
    m.total_p90,
    m.win_prob_p10,
    m.win_prob_p90,
    m.moneyline,
    m.total,
    m.spread,
    m.spread_odds,
    m.our_total,
    m.total_diff,
    m.total_play,
    m.ev_flag,
    m.run_line_ev_flag,
    m.ml_confidence,
    m.run_line_confidence,
    m.high_variance_flag,
    m.kelly_full_ml,
    m.kelly_quarter_ml,
    m.kelly_full_rl,
    m.kelly_quarter_rl,
    m.kelly_full_total,
    m.kelly_quarter_total,
    m.p_cover,
    m.p_over,
    m.p_under,
    m.total_over_odds,
    m.total_under_odds,
    m.lineups_locked,
    m.lineup_source,
    m.prediction_updated_at,
    m.posterior_age_days,
    m.created_at,
    m.updated_at,
    m.lineup_hash,
    m.start_time,
    'v2'::text AS model_version,
    g2.status AS game_status,
    g2.home_team,
    g2.away_team,
    g2.home_score,
    g2.away_score
   FROM (mlb.model_outputs_season m
     JOIN mlb.games g2 ON (((g2.game_pk = m.game_pk) AND (g2.game_date = (m.date)::date))))
  WHERE ((m.date)::date >= '2026-05-12'::date);


--
-- Name: model_outputs_v1_archive; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.model_outputs_v1_archive (
    game_pk bigint,
    date text,
    team text,
    starter text,
    expected_runs real,
    win_prob double precision,
    our_odds bigint,
    is_home bigint,
    moneyline double precision,
    total double precision,
    spread double precision,
    spread_odds double precision,
    total_over_odds bigint,
    total_under_odds bigint,
    our_total real,
    total_diff double precision,
    p_cover double precision,
    p_over double precision,
    p_under double precision,
    total_play text,
    ev_flag text,
    run_line_ev_flag text,
    kelly_full_ml double precision,
    kelly_quarter_ml double precision,
    kelly_full_rl double precision,
    kelly_quarter_rl double precision,
    kelly_full_total double precision,
    kelly_quarter_total double precision,
    ml_confidence double precision,
    run_line_confidence double precision,
    high_variance_flag text
);


--
-- Name: odds; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.odds (
    id integer NOT NULL,
    game_pk integer,
    team character varying(3) NOT NULL,
    book character varying(30) NOT NULL,
    moneyline integer,
    spread numeric(3,1),
    spread_odds integer,
    total numeric(4,1),
    total_over_odds integer,
    total_under_odds integer,
    scraped_at timestamp with time zone DEFAULT now()
);


--
-- Name: odds_id_seq; Type: SEQUENCE; Schema: mlb; Owner: -
--

CREATE SEQUENCE mlb.odds_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: odds_id_seq; Type: SEQUENCE OWNED BY; Schema: mlb; Owner: -
--

ALTER SEQUENCE mlb.odds_id_seq OWNED BY mlb.odds.id;


--
-- Name: park_factors; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.park_factors (
    team character varying(3) NOT NULL,
    venue character varying(100) NOT NULL,
    season integer DEFAULT 2026 NOT NULL,
    park_factor integer DEFAULT 100 NOT NULL
);


--
-- Name: pitcher_stats; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.pitcher_stats (
    id integer NOT NULL,
    pitcher_name character varying(100) NOT NULL,
    team character varying(3) NOT NULL,
    season integer DEFAULT 2026 NOT NULL,
    role character varying(10) DEFAULT 'starter'::character varying NOT NULL,
    ip numeric(5,1),
    era numeric(5,2),
    fip numeric(5,2),
    xfip numeric(5,2),
    siera numeric(5,2),
    whip numeric(5,3),
    k_9 numeric(5,2),
    bb_9 numeric(5,2),
    hr_9 numeric(5,2),
    pitcher_id integer,
    avg_ip_per_start numeric
);


--
-- Name: pitcher_stats_id_seq; Type: SEQUENCE; Schema: mlb; Owner: -
--

CREATE SEQUENCE mlb.pitcher_stats_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: pitcher_stats_id_seq; Type: SEQUENCE OWNED BY; Schema: mlb; Owner: -
--

ALTER SEQUENCE mlb.pitcher_stats_id_seq OWNED BY mlb.pitcher_stats.id;


--
-- Name: pitcher_workload; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.pitcher_workload (
    game_date date NOT NULL,
    pitcher_id integer NOT NULL,
    team text NOT NULL,
    outs integer NOT NULL,
    role text NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT pitcher_workload_role_check CHECK ((role = ANY (ARRAY['SP'::text, 'RP'::text])))
);


--
-- Name: playoff_forecasts; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.playoff_forecasts (
    season integer NOT NULL,
    generated_at timestamp with time zone NOT NULL,
    payload jsonb NOT NULL,
    stage text NOT NULL,
    CONSTRAINT playoff_forecasts_payload_check CHECK (((payload ->> 'schema_version'::text) = '1'::text)),
    CONSTRAINT playoff_forecasts_season_check CHECK ((season >= 2022)),
    CONSTRAINT playoff_forecasts_stage_check CHECK ((stage = ANY (ARRAY['WC'::text, 'DS'::text, 'CS'::text, 'WS'::text])))
);


--
-- Name: posterior_sigmas; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.posterior_sigmas (
    refit_date date NOT NULL,
    sigma_name text NOT NULL,
    mean numeric NOT NULL,
    p10 numeric,
    p90 numeric
);


--
-- Name: posterior_skills; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.posterior_skills (
    refit_date date NOT NULL,
    actor_type text NOT NULL,
    split_label text NOT NULL,
    rank_type text NOT NULL,
    rank integer NOT NULL,
    actor_id bigint NOT NULL,
    actor_name text,
    team text,
    skill_score numeric NOT NULL,
    CONSTRAINT posterior_skills_actor_type_check CHECK ((actor_type = ANY (ARRAY['batter'::text, 'pitcher'::text]))),
    CONSTRAINT posterior_skills_rank_check CHECK (((rank >= 1) AND (rank <= 10))),
    CONSTRAINT posterior_skills_rank_type_check CHECK ((rank_type = ANY (ARRAY['top'::text, 'bottom'::text])))
);


--
-- Name: probable_starters; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.probable_starters (
    id integer NOT NULL,
    game_pk integer,
    team character varying(3) NOT NULL,
    pitcher_name character varying(100) NOT NULL,
    pitcher_id integer,
    handedness character(1),
    is_home boolean NOT NULL
);


--
-- Name: probable_starters_id_seq; Type: SEQUENCE; Schema: mlb; Owner: -
--

CREATE SEQUENCE mlb.probable_starters_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: probable_starters_id_seq; Type: SEQUENCE OWNED BY; Schema: mlb; Owner: -
--

ALTER SEQUENCE mlb.probable_starters_id_seq OWNED BY mlb.probable_starters.id;


--
-- Name: team_batting; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.team_batting (
    id integer NOT NULL,
    team character varying(3) NOT NULL,
    season integer DEFAULT 2026 NOT NULL,
    split character varying(10) NOT NULL,
    pa integer,
    wrc_plus numeric(6,1),
    woba numeric(5,3),
    ops numeric(5,3),
    slg numeric(5,3),
    obp numeric(5,3),
    iso numeric(5,3),
    babip numeric(5,3),
    k_pct numeric(5,1),
    bb_pct numeric(5,1)
);


--
-- Name: team_batting_id_seq; Type: SEQUENCE; Schema: mlb; Owner: -
--

CREATE SEQUENCE mlb.team_batting_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: team_batting_id_seq; Type: SEQUENCE OWNED BY; Schema: mlb; Owner: -
--

ALTER SEQUENCE mlb.team_batting_id_seq OWNED BY mlb.team_batting.id;


--
-- Name: weather; Type: TABLE; Schema: mlb; Owner: -
--

CREATE TABLE mlb.weather (
    game_pk integer NOT NULL,
    wind_speed_mph integer,
    wind_dir_raw text,
    wind_dir_enum text,
    wind_out_component numeric,
    temp_f integer,
    condition text,
    is_dome boolean DEFAULT false NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: bullpen_stats id; Type: DEFAULT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.bullpen_stats ALTER COLUMN id SET DEFAULT nextval('mlb.bullpen_stats_id_seq'::regclass);


--
-- Name: experiment_runs id; Type: DEFAULT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.experiment_runs ALTER COLUMN id SET DEFAULT nextval('mlb.experiment_runs_id_seq'::regclass);


--
-- Name: model_calibration id; Type: DEFAULT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.model_calibration ALTER COLUMN id SET DEFAULT nextval('mlb.model_calibration_id_seq'::regclass);


--
-- Name: model_edge_buckets id; Type: DEFAULT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.model_edge_buckets ALTER COLUMN id SET DEFAULT nextval('mlb.model_edge_buckets_id_seq'::regclass);


--
-- Name: model_evaluation id; Type: DEFAULT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.model_evaluation ALTER COLUMN id SET DEFAULT nextval('mlb.model_evaluation_id_seq'::regclass);


--
-- Name: model_feature_importance id; Type: DEFAULT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.model_feature_importance ALTER COLUMN id SET DEFAULT nextval('mlb.model_feature_importance_id_seq'::regclass);


--
-- Name: odds id; Type: DEFAULT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.odds ALTER COLUMN id SET DEFAULT nextval('mlb.odds_id_seq'::regclass);


--
-- Name: pitcher_stats id; Type: DEFAULT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.pitcher_stats ALTER COLUMN id SET DEFAULT nextval('mlb.pitcher_stats_id_seq'::regclass);


--
-- Name: probable_starters id; Type: DEFAULT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.probable_starters ALTER COLUMN id SET DEFAULT nextval('mlb.probable_starters_id_seq'::regclass);


--
-- Name: team_batting id; Type: DEFAULT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.team_batting ALTER COLUMN id SET DEFAULT nextval('mlb.team_batting_id_seq'::regclass);


--
-- Name: bullpen_daily bullpen_daily_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.bullpen_daily
    ADD CONSTRAINT bullpen_daily_pkey PRIMARY KEY (game_date, team);


--
-- Name: bullpen_stats bullpen_stats_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.bullpen_stats
    ADD CONSTRAINT bullpen_stats_pkey PRIMARY KEY (id);


--
-- Name: bullpen_stats bullpen_stats_team_season_key; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.bullpen_stats
    ADD CONSTRAINT bullpen_stats_team_season_key UNIQUE (team, season);


--
-- Name: experiment_runs experiment_runs_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.experiment_runs
    ADD CONSTRAINT experiment_runs_pkey PRIMARY KEY (id);


--
-- Name: games games_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.games
    ADD CONSTRAINT games_pkey PRIMARY KEY (game_pk);


--
-- Name: live_win_probability live_win_probability_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.live_win_probability
    ADD CONSTRAINT live_win_probability_pkey PRIMARY KEY (game_pk);


--
-- Name: model_calibration model_calibration_date_bin_mid_key; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.model_calibration
    ADD CONSTRAINT model_calibration_date_bin_mid_key UNIQUE (date, bin_mid);


--
-- Name: model_calibration model_calibration_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.model_calibration
    ADD CONSTRAINT model_calibration_pkey PRIMARY KEY (id);


--
-- Name: model_edge_buckets model_edge_buckets_date_eval_window_bucket_label_key; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.model_edge_buckets
    ADD CONSTRAINT model_edge_buckets_date_eval_window_bucket_label_key UNIQUE (date, eval_window, bucket_label);


--
-- Name: model_edge_buckets model_edge_buckets_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.model_edge_buckets
    ADD CONSTRAINT model_edge_buckets_pkey PRIMARY KEY (id);


--
-- Name: model_evaluation model_evaluation_date_window_key; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.model_evaluation
    ADD CONSTRAINT model_evaluation_date_window_key UNIQUE (date, eval_window);


--
-- Name: model_evaluation model_evaluation_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.model_evaluation
    ADD CONSTRAINT model_evaluation_pkey PRIMARY KEY (id);


--
-- Name: model_feature_importance model_feature_importance_date_feature_key; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.model_feature_importance
    ADD CONSTRAINT model_feature_importance_date_feature_key UNIQUE (date, feature);


--
-- Name: model_feature_importance model_feature_importance_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.model_feature_importance
    ADD CONSTRAINT model_feature_importance_pkey PRIMARY KEY (id);


--
-- Name: model_outputs model_outputs_pk; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.model_outputs
    ADD CONSTRAINT model_outputs_pk PRIMARY KEY (game_pk, team);


--
-- Name: model_outputs_season_v1_archive model_outputs_season_game_pk_team_unique; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.model_outputs_season_v1_archive
    ADD CONSTRAINT model_outputs_season_game_pk_team_unique UNIQUE (game_pk, team);


--
-- Name: model_outputs_season model_outputs_season_v2_pk; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.model_outputs_season
    ADD CONSTRAINT model_outputs_season_v2_pk UNIQUE (game_pk, team);


--
-- Name: odds odds_game_pk_team_book_key; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.odds
    ADD CONSTRAINT odds_game_pk_team_book_key UNIQUE (game_pk, team, book);


--
-- Name: odds odds_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.odds
    ADD CONSTRAINT odds_pkey PRIMARY KEY (id);


--
-- Name: park_factors park_factors_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.park_factors
    ADD CONSTRAINT park_factors_pkey PRIMARY KEY (team, season);


--
-- Name: pitcher_stats pitcher_stats_pitcher_name_team_season_role_key; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.pitcher_stats
    ADD CONSTRAINT pitcher_stats_pitcher_name_team_season_role_key UNIQUE (pitcher_name, team, season, role);


--
-- Name: pitcher_stats pitcher_stats_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.pitcher_stats
    ADD CONSTRAINT pitcher_stats_pkey PRIMARY KEY (id);


--
-- Name: pitcher_workload pitcher_workload_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.pitcher_workload
    ADD CONSTRAINT pitcher_workload_pkey PRIMARY KEY (game_date, pitcher_id);


--
-- Name: playoff_forecasts playoff_forecasts_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.playoff_forecasts
    ADD CONSTRAINT playoff_forecasts_pkey PRIMARY KEY (season, stage);


--
-- Name: posterior_sigmas posterior_sigmas_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.posterior_sigmas
    ADD CONSTRAINT posterior_sigmas_pkey PRIMARY KEY (refit_date, sigma_name);


--
-- Name: posterior_skills posterior_skills_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.posterior_skills
    ADD CONSTRAINT posterior_skills_pkey PRIMARY KEY (refit_date, actor_type, split_label, rank_type, rank);


--
-- Name: probable_starters probable_starters_game_pk_team_key; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.probable_starters
    ADD CONSTRAINT probable_starters_game_pk_team_key UNIQUE (game_pk, team);


--
-- Name: probable_starters probable_starters_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.probable_starters
    ADD CONSTRAINT probable_starters_pkey PRIMARY KEY (id);


--
-- Name: team_batting team_batting_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.team_batting
    ADD CONSTRAINT team_batting_pkey PRIMARY KEY (id);


--
-- Name: team_batting team_batting_team_season_split_key; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.team_batting
    ADD CONSTRAINT team_batting_team_season_split_key UNIQUE (team, season, split);


--
-- Name: weather weather_pkey; Type: CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.weather
    ADD CONSTRAINT weather_pkey PRIMARY KEY (game_pk);


--
-- Name: idx_bullpen_daily_team_date; Type: INDEX; Schema: mlb; Owner: -
--

CREATE INDEX idx_bullpen_daily_team_date ON mlb.bullpen_daily USING btree (team, game_date);


--
-- Name: idx_games_date; Type: INDEX; Schema: mlb; Owner: -
--

CREATE INDEX idx_games_date ON mlb.games USING btree (game_date);


--
-- Name: idx_games_status; Type: INDEX; Schema: mlb; Owner: -
--

CREATE INDEX idx_games_status ON mlb.games USING btree (status);


--
-- Name: idx_model_outputs_season_start_time; Type: INDEX; Schema: mlb; Owner: -
--

CREATE INDEX idx_model_outputs_season_start_time ON mlb.model_outputs_season USING btree (start_time DESC);


--
-- Name: idx_model_outputs_start_time; Type: INDEX; Schema: mlb; Owner: -
--

CREATE INDEX idx_model_outputs_start_time ON mlb.model_outputs USING btree (start_time DESC);


--
-- Name: idx_odds_game_pk; Type: INDEX; Schema: mlb; Owner: -
--

CREATE INDEX idx_odds_game_pk ON mlb.odds USING btree (game_pk);


--
-- Name: idx_pitcher_workload_pitcher_date; Type: INDEX; Schema: mlb; Owner: -
--

CREATE INDEX idx_pitcher_workload_pitcher_date ON mlb.pitcher_workload USING btree (pitcher_id, game_date);


--
-- Name: idx_pitcher_workload_team_date; Type: INDEX; Schema: mlb; Owner: -
--

CREATE INDEX idx_pitcher_workload_team_date ON mlb.pitcher_workload USING btree (team, game_date);


--
-- Name: idx_posterior_sigmas_lookup; Type: INDEX; Schema: mlb; Owner: -
--

CREATE INDEX idx_posterior_sigmas_lookup ON mlb.posterior_sigmas USING btree (refit_date DESC, sigma_name);


--
-- Name: idx_posterior_skills_lookup; Type: INDEX; Schema: mlb; Owner: -
--

CREATE INDEX idx_posterior_skills_lookup ON mlb.posterior_skills USING btree (refit_date DESC, actor_type, split_label, rank_type, rank);


--
-- Name: idx_probable_starters_game_pk; Type: INDEX; Schema: mlb; Owner: -
--

CREATE INDEX idx_probable_starters_game_pk ON mlb.probable_starters USING btree (game_pk);


--
-- Name: model_outputs_season_v2_date_idx; Type: INDEX; Schema: mlb; Owner: -
--

CREATE INDEX model_outputs_season_v2_date_idx ON mlb.model_outputs_season USING btree (date);


--
-- Name: model_outputs_season_v2_game_pk_idx; Type: INDEX; Schema: mlb; Owner: -
--

CREATE INDEX model_outputs_season_v2_game_pk_idx ON mlb.model_outputs_season USING btree (game_pk);


--
-- Name: model_outputs_v2_date_idx; Type: INDEX; Schema: mlb; Owner: -
--

CREATE INDEX model_outputs_v2_date_idx ON mlb.model_outputs USING btree (date);


--
-- Name: live_win_probability live_win_probability_revalidate; Type: TRIGGER; Schema: mlb; Owner: -
--

CREATE TRIGGER live_win_probability_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.live_win_probability FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: playoff_forecasts playoff_forecasts_revalidate; Type: TRIGGER; Schema: mlb; Owner: -
--

CREATE TRIGGER playoff_forecasts_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.playoff_forecasts FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: games site_revalidate; Type: TRIGGER; Schema: mlb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.games FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: model_calibration site_revalidate; Type: TRIGGER; Schema: mlb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.model_calibration FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: model_edge_buckets site_revalidate; Type: TRIGGER; Schema: mlb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.model_edge_buckets FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: model_evaluation site_revalidate; Type: TRIGGER; Schema: mlb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.model_evaluation FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: model_outputs site_revalidate; Type: TRIGGER; Schema: mlb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.model_outputs FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: model_outputs_season site_revalidate; Type: TRIGGER; Schema: mlb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.model_outputs_season FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: model_outputs_season_v1_archive site_revalidate; Type: TRIGGER; Schema: mlb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.model_outputs_season_v1_archive FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: posterior_sigmas site_revalidate; Type: TRIGGER; Schema: mlb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.posterior_sigmas FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: posterior_skills site_revalidate; Type: TRIGGER; Schema: mlb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.posterior_skills FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: model_outputs_season_v1_archive trg_model_outputs_season_updated_at; Type: TRIGGER; Schema: mlb; Owner: -
--

CREATE TRIGGER trg_model_outputs_season_updated_at BEFORE UPDATE ON mlb.model_outputs_season_v1_archive FOR EACH ROW EXECUTE FUNCTION mlb.set_updated_at_model_outputs_season();


--
-- Name: odds odds_game_pk_fkey; Type: FK CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.odds
    ADD CONSTRAINT odds_game_pk_fkey FOREIGN KEY (game_pk) REFERENCES mlb.games(game_pk) ON DELETE CASCADE;


--
-- Name: probable_starters probable_starters_game_pk_fkey; Type: FK CONSTRAINT; Schema: mlb; Owner: -
--

ALTER TABLE ONLY mlb.probable_starters
    ADD CONSTRAINT probable_starters_game_pk_fkey FOREIGN KEY (game_pk) REFERENCES mlb.games(game_pk) ON DELETE CASCADE;


--
-- Name: bullpen_daily; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.bullpen_daily ENABLE ROW LEVEL SECURITY;

--
-- Name: bullpen_stats; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.bullpen_stats ENABLE ROW LEVEL SECURITY;

--
-- Name: experiment_runs; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.experiment_runs ENABLE ROW LEVEL SECURITY;

--
-- Name: games; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.games ENABLE ROW LEVEL SECURITY;

--
-- Name: live_win_probability; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.live_win_probability ENABLE ROW LEVEL SECURITY;

--
-- Name: live_win_probability live_win_probability_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY live_win_probability_read ON mlb.live_win_probability FOR SELECT USING (true);


--
-- Name: model_calibration; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.model_calibration ENABLE ROW LEVEL SECURITY;

--
-- Name: model_edge_buckets; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.model_edge_buckets ENABLE ROW LEVEL SECURITY;

--
-- Name: model_evaluation; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.model_evaluation ENABLE ROW LEVEL SECURITY;

--
-- Name: model_feature_importance; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.model_feature_importance ENABLE ROW LEVEL SECURITY;

--
-- Name: model_outputs; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.model_outputs ENABLE ROW LEVEL SECURITY;

--
-- Name: model_outputs_season; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.model_outputs_season ENABLE ROW LEVEL SECURITY;

--
-- Name: model_outputs_season_v1_archive; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.model_outputs_season_v1_archive ENABLE ROW LEVEL SECURITY;

--
-- Name: model_outputs_v1_archive; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.model_outputs_v1_archive ENABLE ROW LEVEL SECURITY;

--
-- Name: odds; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.odds ENABLE ROW LEVEL SECURITY;

--
-- Name: park_factors; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.park_factors ENABLE ROW LEVEL SECURITY;

--
-- Name: pitcher_stats; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.pitcher_stats ENABLE ROW LEVEL SECURITY;

--
-- Name: pitcher_workload; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.pitcher_workload ENABLE ROW LEVEL SECURITY;

--
-- Name: playoff_forecasts; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.playoff_forecasts ENABLE ROW LEVEL SECURITY;

--
-- Name: playoff_forecasts playoff_forecasts_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY playoff_forecasts_read ON mlb.playoff_forecasts FOR SELECT USING (true);


--
-- Name: posterior_sigmas; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.posterior_sigmas ENABLE ROW LEVEL SECURITY;

--
-- Name: posterior_skills; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.posterior_skills ENABLE ROW LEVEL SECURITY;

--
-- Name: probable_starters; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.probable_starters ENABLE ROW LEVEL SECURITY;

--
-- Name: bullpen_daily public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.bullpen_daily FOR SELECT TO authenticated, anon USING (true);


--
-- Name: bullpen_stats public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.bullpen_stats FOR SELECT TO authenticated, anon USING (true);


--
-- Name: experiment_runs public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.experiment_runs FOR SELECT TO authenticated, anon USING (true);


--
-- Name: games public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.games FOR SELECT TO authenticated, anon USING (true);


--
-- Name: model_calibration public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.model_calibration FOR SELECT TO authenticated, anon USING (true);


--
-- Name: model_edge_buckets public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.model_edge_buckets FOR SELECT TO authenticated, anon USING (true);


--
-- Name: model_evaluation public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.model_evaluation FOR SELECT TO authenticated, anon USING (true);


--
-- Name: model_feature_importance public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.model_feature_importance FOR SELECT TO authenticated, anon USING (true);


--
-- Name: model_outputs public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.model_outputs FOR SELECT TO authenticated, anon USING (true);


--
-- Name: model_outputs_season public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.model_outputs_season FOR SELECT TO authenticated, anon USING (true);


--
-- Name: model_outputs_season_v1_archive public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.model_outputs_season_v1_archive FOR SELECT TO authenticated, anon USING (true);


--
-- Name: model_outputs_v1_archive public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.model_outputs_v1_archive FOR SELECT TO authenticated, anon USING (true);


--
-- Name: odds public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.odds FOR SELECT TO authenticated, anon USING (true);


--
-- Name: park_factors public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.park_factors FOR SELECT TO authenticated, anon USING (true);


--
-- Name: pitcher_stats public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.pitcher_stats FOR SELECT TO authenticated, anon USING (true);


--
-- Name: pitcher_workload public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.pitcher_workload FOR SELECT TO authenticated, anon USING (true);


--
-- Name: posterior_sigmas public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.posterior_sigmas FOR SELECT TO authenticated, anon USING (true);


--
-- Name: posterior_skills public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.posterior_skills FOR SELECT TO authenticated, anon USING (true);


--
-- Name: probable_starters public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.probable_starters FOR SELECT TO authenticated, anon USING (true);


--
-- Name: team_batting public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.team_batting FOR SELECT TO authenticated, anon USING (true);


--
-- Name: weather public_read; Type: POLICY; Schema: mlb; Owner: -
--

CREATE POLICY public_read ON mlb.weather FOR SELECT TO authenticated, anon USING (true);


--
-- Name: team_batting; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.team_batting ENABLE ROW LEVEL SECURITY;

--
-- Name: weather; Type: ROW SECURITY; Schema: mlb; Owner: -
--

ALTER TABLE mlb.weather ENABLE ROW LEVEL SECURITY;

--
-- Name: SCHEMA mlb; Type: ACL; Schema: -; Owner: -
--

GRANT USAGE ON SCHEMA mlb TO anon;
GRANT USAGE ON SCHEMA mlb TO authenticated;
GRANT USAGE ON SCHEMA mlb TO service_role;


--
-- Name: FUNCTION bet_record_summary(p_from text, p_team text); Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON FUNCTION mlb.bet_record_summary(p_from text, p_team text) TO anon;
GRANT ALL ON FUNCTION mlb.bet_record_summary(p_from text, p_team text) TO authenticated;
GRANT ALL ON FUNCTION mlb.bet_record_summary(p_from text, p_team text) TO service_role;


--
-- Name: FUNCTION set_updated_at_model_outputs_season(); Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON FUNCTION mlb.set_updated_at_model_outputs_season() TO anon;
GRANT ALL ON FUNCTION mlb.set_updated_at_model_outputs_season() TO authenticated;
GRANT ALL ON FUNCTION mlb.set_updated_at_model_outputs_season() TO service_role;


--
-- Name: TABLE games; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.games TO anon;
GRANT ALL ON TABLE mlb.games TO authenticated;
GRANT ALL ON TABLE mlb.games TO service_role;


--
-- Name: TABLE model_outputs_season; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.model_outputs_season TO anon;
GRANT ALL ON TABLE mlb.model_outputs_season TO authenticated;
GRANT ALL ON TABLE mlb.model_outputs_season TO service_role;


--
-- Name: TABLE model_outputs_season_v1_archive; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.model_outputs_season_v1_archive TO anon;
GRANT ALL ON TABLE mlb.model_outputs_season_v1_archive TO authenticated;
GRANT ALL ON TABLE mlb.model_outputs_season_v1_archive TO service_role;


--
-- Name: TABLE bet_ledger_v; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.bet_ledger_v TO anon;
GRANT ALL ON TABLE mlb.bet_ledger_v TO authenticated;
GRANT ALL ON TABLE mlb.bet_ledger_v TO service_role;


--
-- Name: TABLE bet_ledger_agg_v; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.bet_ledger_agg_v TO anon;
GRANT ALL ON TABLE mlb.bet_ledger_agg_v TO authenticated;
GRANT ALL ON TABLE mlb.bet_ledger_agg_v TO service_role;


--
-- Name: TABLE bullpen_daily; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.bullpen_daily TO anon;
GRANT ALL ON TABLE mlb.bullpen_daily TO authenticated;
GRANT ALL ON TABLE mlb.bullpen_daily TO service_role;


--
-- Name: TABLE bullpen_stats; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.bullpen_stats TO anon;
GRANT ALL ON TABLE mlb.bullpen_stats TO authenticated;
GRANT ALL ON TABLE mlb.bullpen_stats TO service_role;


--
-- Name: SEQUENCE bullpen_stats_id_seq; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON SEQUENCE mlb.bullpen_stats_id_seq TO anon;
GRANT ALL ON SEQUENCE mlb.bullpen_stats_id_seq TO authenticated;
GRANT ALL ON SEQUENCE mlb.bullpen_stats_id_seq TO service_role;


--
-- Name: TABLE experiment_runs; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.experiment_runs TO anon;
GRANT ALL ON TABLE mlb.experiment_runs TO authenticated;
GRANT ALL ON TABLE mlb.experiment_runs TO service_role;


--
-- Name: SEQUENCE experiment_runs_id_seq; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON SEQUENCE mlb.experiment_runs_id_seq TO anon;
GRANT ALL ON SEQUENCE mlb.experiment_runs_id_seq TO authenticated;
GRANT ALL ON SEQUENCE mlb.experiment_runs_id_seq TO service_role;


--
-- Name: TABLE live_win_probability; Type: ACL; Schema: mlb; Owner: -
--

GRANT SELECT ON TABLE mlb.live_win_probability TO anon;
GRANT SELECT ON TABLE mlb.live_win_probability TO authenticated;
GRANT ALL ON TABLE mlb.live_win_probability TO service_role;


--
-- Name: TABLE model_calibration; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.model_calibration TO anon;
GRANT ALL ON TABLE mlb.model_calibration TO authenticated;
GRANT ALL ON TABLE mlb.model_calibration TO service_role;


--
-- Name: SEQUENCE model_calibration_id_seq; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON SEQUENCE mlb.model_calibration_id_seq TO anon;
GRANT ALL ON SEQUENCE mlb.model_calibration_id_seq TO authenticated;
GRANT ALL ON SEQUENCE mlb.model_calibration_id_seq TO service_role;


--
-- Name: TABLE model_edge_buckets; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.model_edge_buckets TO anon;
GRANT ALL ON TABLE mlb.model_edge_buckets TO authenticated;
GRANT ALL ON TABLE mlb.model_edge_buckets TO service_role;


--
-- Name: SEQUENCE model_edge_buckets_id_seq; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON SEQUENCE mlb.model_edge_buckets_id_seq TO anon;
GRANT ALL ON SEQUENCE mlb.model_edge_buckets_id_seq TO authenticated;
GRANT ALL ON SEQUENCE mlb.model_edge_buckets_id_seq TO service_role;


--
-- Name: TABLE model_evaluation; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.model_evaluation TO anon;
GRANT ALL ON TABLE mlb.model_evaluation TO authenticated;
GRANT ALL ON TABLE mlb.model_evaluation TO service_role;


--
-- Name: SEQUENCE model_evaluation_id_seq; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON SEQUENCE mlb.model_evaluation_id_seq TO anon;
GRANT ALL ON SEQUENCE mlb.model_evaluation_id_seq TO authenticated;
GRANT ALL ON SEQUENCE mlb.model_evaluation_id_seq TO service_role;


--
-- Name: TABLE model_feature_importance; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.model_feature_importance TO anon;
GRANT ALL ON TABLE mlb.model_feature_importance TO authenticated;
GRANT ALL ON TABLE mlb.model_feature_importance TO service_role;


--
-- Name: SEQUENCE model_feature_importance_id_seq; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON SEQUENCE mlb.model_feature_importance_id_seq TO anon;
GRANT ALL ON SEQUENCE mlb.model_feature_importance_id_seq TO authenticated;
GRANT ALL ON SEQUENCE mlb.model_feature_importance_id_seq TO service_role;


--
-- Name: TABLE model_outputs; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.model_outputs TO anon;
GRANT ALL ON TABLE mlb.model_outputs TO authenticated;
GRANT ALL ON TABLE mlb.model_outputs TO service_role;


--
-- Name: TABLE model_outputs_season_unified; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.model_outputs_season_unified TO anon;
GRANT ALL ON TABLE mlb.model_outputs_season_unified TO authenticated;
GRANT ALL ON TABLE mlb.model_outputs_season_unified TO service_role;


--
-- Name: TABLE model_outputs_v1_archive; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.model_outputs_v1_archive TO anon;
GRANT ALL ON TABLE mlb.model_outputs_v1_archive TO authenticated;
GRANT ALL ON TABLE mlb.model_outputs_v1_archive TO service_role;


--
-- Name: TABLE odds; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.odds TO anon;
GRANT ALL ON TABLE mlb.odds TO authenticated;
GRANT ALL ON TABLE mlb.odds TO service_role;


--
-- Name: SEQUENCE odds_id_seq; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON SEQUENCE mlb.odds_id_seq TO anon;
GRANT ALL ON SEQUENCE mlb.odds_id_seq TO authenticated;
GRANT ALL ON SEQUENCE mlb.odds_id_seq TO service_role;


--
-- Name: TABLE park_factors; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.park_factors TO anon;
GRANT ALL ON TABLE mlb.park_factors TO authenticated;
GRANT ALL ON TABLE mlb.park_factors TO service_role;


--
-- Name: TABLE pitcher_stats; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.pitcher_stats TO anon;
GRANT ALL ON TABLE mlb.pitcher_stats TO authenticated;
GRANT ALL ON TABLE mlb.pitcher_stats TO service_role;


--
-- Name: SEQUENCE pitcher_stats_id_seq; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON SEQUENCE mlb.pitcher_stats_id_seq TO anon;
GRANT ALL ON SEQUENCE mlb.pitcher_stats_id_seq TO authenticated;
GRANT ALL ON SEQUENCE mlb.pitcher_stats_id_seq TO service_role;


--
-- Name: TABLE pitcher_workload; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.pitcher_workload TO anon;
GRANT ALL ON TABLE mlb.pitcher_workload TO authenticated;
GRANT ALL ON TABLE mlb.pitcher_workload TO service_role;


--
-- Name: TABLE playoff_forecasts; Type: ACL; Schema: mlb; Owner: -
--

GRANT SELECT ON TABLE mlb.playoff_forecasts TO anon;
GRANT SELECT ON TABLE mlb.playoff_forecasts TO authenticated;
GRANT ALL ON TABLE mlb.playoff_forecasts TO service_role;


--
-- Name: TABLE posterior_sigmas; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.posterior_sigmas TO anon;
GRANT ALL ON TABLE mlb.posterior_sigmas TO authenticated;
GRANT ALL ON TABLE mlb.posterior_sigmas TO service_role;


--
-- Name: TABLE posterior_skills; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.posterior_skills TO anon;
GRANT ALL ON TABLE mlb.posterior_skills TO authenticated;
GRANT ALL ON TABLE mlb.posterior_skills TO service_role;


--
-- Name: TABLE probable_starters; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.probable_starters TO anon;
GRANT ALL ON TABLE mlb.probable_starters TO authenticated;
GRANT ALL ON TABLE mlb.probable_starters TO service_role;


--
-- Name: SEQUENCE probable_starters_id_seq; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON SEQUENCE mlb.probable_starters_id_seq TO anon;
GRANT ALL ON SEQUENCE mlb.probable_starters_id_seq TO authenticated;
GRANT ALL ON SEQUENCE mlb.probable_starters_id_seq TO service_role;


--
-- Name: TABLE team_batting; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.team_batting TO anon;
GRANT ALL ON TABLE mlb.team_batting TO authenticated;
GRANT ALL ON TABLE mlb.team_batting TO service_role;


--
-- Name: SEQUENCE team_batting_id_seq; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON SEQUENCE mlb.team_batting_id_seq TO anon;
GRANT ALL ON SEQUENCE mlb.team_batting_id_seq TO authenticated;
GRANT ALL ON SEQUENCE mlb.team_batting_id_seq TO service_role;


--
-- Name: TABLE weather; Type: ACL; Schema: mlb; Owner: -
--

GRANT ALL ON TABLE mlb.weather TO anon;
GRANT ALL ON TABLE mlb.weather TO authenticated;
GRANT ALL ON TABLE mlb.weather TO service_role;


--
-- Name: DEFAULT PRIVILEGES FOR SEQUENCES; Type: DEFAULT ACL; Schema: mlb; Owner: -
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA mlb GRANT SELECT,USAGE ON SEQUENCES TO anon;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA mlb GRANT SELECT,USAGE ON SEQUENCES TO authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA mlb GRANT ALL ON SEQUENCES TO service_role;


--
-- Name: DEFAULT PRIVILEGES FOR FUNCTIONS; Type: DEFAULT ACL; Schema: mlb; Owner: -
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA mlb GRANT ALL ON FUNCTIONS TO anon;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA mlb GRANT ALL ON FUNCTIONS TO authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA mlb GRANT ALL ON FUNCTIONS TO service_role;


--
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: mlb; Owner: -
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA mlb GRANT SELECT ON TABLES TO anon;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA mlb GRANT SELECT ON TABLES TO authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA mlb GRANT ALL ON TABLES TO service_role;


--
-- PostgreSQL database dump complete
--

\unrestrict SL8fUNpNYBGbzkVY1pJrGMqsqw6e6jedy8EKkEHkgv7vMfcjFLemdcmzHFji6SY

