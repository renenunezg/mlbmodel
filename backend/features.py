"""Data loading and feature engineering for the MLB expected-runs model.

Pulls from Supabase tables (probable_starters, pitcher_stats, bullpen_stats,
team_batting, park_factors, games) and produces a single DataFrame with the
12 model features blended across starter/bullpen handedness.
"""
import pandas as pd
import numpy as np
from backend.db import engine


# Dynamic starter/bullpen inning-share constants. See load_training_data for usage.
LEAGUE_STARTER_IP = 5.2
LEAGUE_STARTER_SHARE = round(LEAGUE_STARTER_IP / 9, 3)  # ~0.578
LEAGUE_BULLPEN_RHP_SHARE = 0.6
STARTER_SHARE_MIN, STARTER_SHARE_MAX = 0.35, 0.78


# League-average fallbacks for early season when stats tables are empty
LEAGUE_AVG = {
    "xfip": 4.20, "whip": 1.30, "xfip_bullpen": 4.10, "bullpen_k_9": 9.0,
    "batting_ops": 0.720, "batting_iso": 0.160, "batting_k_pct": 22.0,
    "park_factor": 100, "avg_last5": 4.5, "avg_last10": 4.5, "std_last5": 2.5,
    # Reliever outs in prior 2 days. League norm derived from analysis cache:
    # mean ~17 outs / 2-day window across 826 (date, team) rows.
    "own_bp_outs_2d": 17.0, "opp_bp_outs_2d": 17.0,
}


def compute_starter_inning_share(avg_ip_per_start):
    """avg IP/start / 9, clamped to [0.35, 0.78]. NaN falls back to league mean."""
    if isinstance(avg_ip_per_start, pd.Series):
        share = (pd.to_numeric(avg_ip_per_start, errors="coerce") / 9.0).clip(
            lower=STARTER_SHARE_MIN, upper=STARTER_SHARE_MAX,
        )
        return share.fillna(LEAGUE_STARTER_SHARE)
    if avg_ip_per_start is None or pd.isna(avg_ip_per_start):
        return LEAGUE_STARTER_SHARE
    return float(np.clip(avg_ip_per_start / 9.0, STARTER_SHARE_MIN, STARTER_SHARE_MAX))


def blend_batting_split(vs_r, vs_l, opp_handedness, starter_share, bullpen_rhp_share):
    """Blend a batting split across the starter (known handedness) + bullpen (RHP-share mix)."""
    split_vs_starter = np.where(opp_handedness == "R", vs_r, vs_l)
    split_vs_bullpen = bullpen_rhp_share * vs_r + (1 - bullpen_rhp_share) * vs_l
    return starter_share * split_vs_starter + (1 - starter_share) * split_vs_bullpen


def _safe_read_table(table_name):
    try:
        df = pd.read_sql_table(table_name, con=engine)
        if df.empty:
            print(f"  Warning: {table_name} is empty")
        return df
    except Exception as e:
        print(f"  Warning: could not read {table_name}: {e}")
        return pd.DataFrame()


def load_training_data():
    """Merge all source tables into one DataFrame, filling sparse rows with league averages."""
    print("Loading training data...")
    starters = _safe_read_table("probable_starters")
    if starters.empty:
        print("  ERROR: probable_starters is empty - cannot proceed.")
        return pd.DataFrame()

    # Merge in pitcher stats (xfip, whip, avg_ip_per_start) via pitcher_id (MLB player ID)
    sp_stats = _safe_read_table("pitcher_stats")
    sp_cols = ["pitcher_id", "xfip", "whip"]
    if not sp_stats.empty and "avg_ip_per_start" in sp_stats.columns:
        sp_cols.append("avg_ip_per_start")
    if not sp_stats.empty and "pitcher_id" in sp_stats.columns and "pitcher_id" in starters.columns:
        starters = pd.merge(
            starters,
            sp_stats[sp_cols],
            on="pitcher_id",
            how="left",
        )
    else:
        starters["xfip"] = np.nan
        starters["whip"] = np.nan
        starters["avg_ip_per_start"] = np.nan

    if "avg_ip_per_start" not in starters.columns:
        starters["avg_ip_per_start"] = np.nan

    # Merge in bullpen stats (xfip, k_9, rhp_ip_share)
    bp_stats = _safe_read_table("bullpen_stats")
    bp_cols = ["team", "xfip", "k_9"]
    if not bp_stats.empty and "rhp_ip_share" in bp_stats.columns:
        bp_cols.append("rhp_ip_share")
    if not bp_stats.empty:
        starters = pd.merge(
            starters,
            bp_stats[bp_cols].rename(columns={"xfip": "xfip_bullpen", "k_9": "bullpen_k_9"}),
            on="team",
            how="left",
        )
    else:
        starters["xfip_bullpen"] = np.nan
        starters["bullpen_k_9"] = np.nan
        starters["rhp_ip_share"] = np.nan

    if "rhp_ip_share" not in starters.columns:
        starters["rhp_ip_share"] = np.nan

    # Load batting splits from unified team_batting table
    batting = _safe_read_table("team_batting")
    bat_cols = ["team", "split", "ops", "iso", "k_pct"]
    if not batting.empty:
        batting = batting[[c for c in bat_cols if c in batting.columns]]
        vs_r = batting[batting["split"] == "vs_rhp"].drop(columns=["split"]).rename(
            columns={"ops": "ops_vs_r", "iso": "iso_vs_r", "k_pct": "k_pct_vs_r"}
        )
        vs_l = batting[batting["split"] == "vs_lhp"].drop(columns=["split"]).rename(
            columns={"ops": "ops_vs_l", "iso": "iso_vs_l", "k_pct": "k_pct_vs_l"}
        )
        starters = pd.merge(starters, vs_r, on="team", how="left")
        starters = pd.merge(starters, vs_l, on="team", how="left")
    else:
        for suffix in ["_vs_r", "_vs_l"]:
            for col in ["ops", "iso", "k_pct"]:
                starters[f"{col}{suffix}"] = np.nan

    # Self-merge to get opponent info: pitcher handedness, starter avg IP/start
    # (for dynamic inning-share blend), and opp team's bullpen RHP IP share.
    # Use left join so games with only one known starter aren't dropped - missing
    # opponent handedness defaults to "R" (league-average ~70% of starters are RHP).
    opp = starters[["game_pk", "team", "handedness", "avg_ip_per_start", "rhp_ip_share"]].rename(
        columns={
            "team": "opp_team",
            "handedness": "opp_handedness",
            "avg_ip_per_start": "opp_avg_ip_per_start",
            "rhp_ip_share": "opp_rhp_ip_share",
        }
    )
    starters = pd.merge(starters, opp, on="game_pk", how="left")
    starters = starters[starters["team"] != starters["opp_team"].fillna("")]
    starters["opp_handedness"] = starters["opp_handedness"].fillna("R")

    opp_ip = pd.to_numeric(starters["opp_avg_ip_per_start"], errors="coerce")
    starter_fallback_mask = opp_ip.isna()
    starter_share = compute_starter_inning_share(opp_ip)
    starters["starter_inning_share"] = starter_share

    bullpen_rhp = pd.to_numeric(starters["opp_rhp_ip_share"], errors="coerce")
    bullpen_fallback_mask = bullpen_rhp.isna()
    bullpen_rhp = bullpen_rhp.fillna(LEAGUE_BULLPEN_RHP_SHARE)
    starters["bullpen_rhp_share"] = bullpen_rhp

    if len(starters) > 0:
        ss_frac = float(starter_fallback_mask.mean())
        bp_frac = float(bullpen_fallback_mask.mean())
        if ss_frac > 0.5:
            print(f"  starter_inning_share: {ss_frac:.0%} fallback to league mean")
        if bp_frac > 0.5:
            print(f"  bullpen_rhp_share: {bp_frac:.0%} fallback to league mean")

    # Blend batting splits vs the opposing starter (known handedness) and the
    # opposing bullpen (distribution from opp_rhp_ip_share).
    for stat, vs_r_col, vs_l_col in [
        ("batting_ops", "ops_vs_r", "ops_vs_l"),
        ("batting_iso", "iso_vs_r", "iso_vs_l"),
        ("batting_k_pct", "k_pct_vs_r", "k_pct_vs_l"),
    ]:
        vs_r = starters.get(vs_r_col, pd.Series(np.nan, index=starters.index))
        vs_l = starters.get(vs_l_col, pd.Series(np.nan, index=starters.index))
        starters[stat] = blend_batting_split(
            vs_r, vs_l, starters["opp_handedness"].values, starter_share, bullpen_rhp,
        )

    # Load park factors and merge based on home team
    parks = _safe_read_table("park_factors")
    if not parks.empty:
        home_teams = starters[starters["is_home"] == True][["game_pk", "team"]].rename(columns={"team": "home_team"})
        starters = pd.merge(starters, home_teams, on="game_pk", how="left")
        home_parks = pd.merge(home_teams, parks, left_on="home_team", right_on="team", how="left")[["game_pk", "home_team", "park_factor"]]
        # Audit: surface silent merge failures (unmapped home team names) as loud warnings.
        unmapped = home_parks[home_parks["park_factor"].isna()]["home_team"].unique().tolist()
        if unmapped:
            print(f"  WARNING: park_factors missing for home teams {unmapped} - check team_mappings.py")
        starters = pd.merge(starters, home_parks.drop(columns=["home_team"]), on="game_pk", how="left")
    else:
        starters["park_factor"] = 100

    # Load completed games
    games = _safe_read_table("games")
    if not games.empty:
        games = games[games["status"] == "Final"]
        games = games.dropna(subset=["away_score", "home_score"])

    if games.empty:
        print("  Warning: no completed games - rolling averages will use league average")
        starters["game_date"] = pd.NaT
        starters["actual_runs"] = np.nan
        starters["avg_last5"] = np.nan
        starters["avg_last10"] = np.nan
        starters["std_last5"] = np.nan
    else:
        starters = pd.merge(
            starters,
            games[["game_pk", "game_date", "away_team", "away_score", "home_team", "home_score"]].rename(
                columns={"home_team": "game_home_team", "away_team": "game_away_team"}
            ),
            on="game_pk",
            how="left",
        )

        starters["actual_runs"] = np.where(
            starters["is_home"] == True,
            starters["home_score"],
            starters["away_score"],
        )

        # Rolling team run averages - strictly prior dates only. Games sharing a
        # (team, game_date) inherit the window computed as of the first game on that date,
        # so doubleheader game 2 never leaks its own result into its features.
        games["game_date"] = pd.to_datetime(games["game_date"])
        starters["game_date"] = pd.to_datetime(starters["game_date"])

        home_runs = games[["game_pk", "game_date", "home_team", "home_score"]].rename(
            columns={"home_team": "team", "home_score": "runs"}
        )
        away_runs = games[["game_pk", "game_date", "away_team", "away_score"]].rename(
            columns={"away_team": "team", "away_score": "runs"}
        )
        team_games = pd.concat([home_runs, away_runs], ignore_index=True)
        team_games = team_games.sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)

        # Shift by 1 within each team so the window excludes the current row's runs
        team_games["runs_shifted"] = team_games.groupby("team")["runs"].shift(1)

        grp = team_games.groupby("team")["runs_shifted"]
        team_games["avg_last5"] = grp.rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        team_games["avg_last10"] = grp.rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
        team_games["std_last5"] = grp.rolling(5, min_periods=2).std().reset_index(level=0, drop=True)

        # Doubleheader handling: all games on the same date share the FIRST game's rolling,
        # so neither game sees the other's runs (matches old `game_date < current_date`).
        for col in ["avg_last5", "avg_last10", "std_last5"]:
            team_games[col] = team_games.groupby(["team", "game_date"])[col].transform("first")

        starters = pd.merge(
            starters,
            team_games[["game_pk", "team", "avg_last5", "avg_last10", "std_last5"]],
            on=["game_pk", "team"],
            how="left",
        )

    # Bullpen rest features: reliever outs in prior 2 days, for own team and opponent.
    # Source: bullpen_daily table (one row per (date, team), reliever_outs aggregated
    # from MLB boxscores). The lookback is strictly prior - game date itself is
    # excluded so we never leak today's bullpen usage into a feature.
    bp_daily = _safe_read_table("bullpen_daily")
    if not bp_daily.empty and "game_date" in starters.columns:
        bp_daily["game_date"] = pd.to_datetime(bp_daily["game_date"])
        starters["game_date"] = pd.to_datetime(starters["game_date"])

        def _prior_outs(team_col: str, label: str, days: int = 2) -> pd.Series:
            out = []
            bp_idx = bp_daily.set_index(["team", "game_date"])["reliever_outs"]
            for _, r in starters.iterrows():
                t = r[team_col]
                d = r["game_date"]
                if pd.isna(t) or pd.isna(d):
                    out.append(np.nan); continue
                window_dates = [d - pd.Timedelta(days=k) for k in range(1, days + 1)]
                total = 0; any_match = False
                for wd in window_dates:
                    try:
                        total += int(bp_idx.loc[(t, wd)]); any_match = True
                    except KeyError:
                        pass
                # If a team had ZERO games in window (legit off-day), 0 is correct.
                # If we have NO bullpen_daily rows for that team at all, fall back to NaN.
                if not any_match and (t not in bp_daily["team"].values):
                    out.append(np.nan)
                else:
                    out.append(total)
            return pd.Series(out, index=starters.index, dtype=float)

        # Need opponent team per row. Use game_home/away from earlier merge if present;
        # otherwise derive from the games table.
        if "game_home_team" in starters.columns and "game_away_team" in starters.columns:
            starters["opp_team"] = np.where(
                starters["is_home"] == True,
                starters["game_away_team"],
                starters["game_home_team"],
            )
        else:
            games_lookup = _safe_read_table("games")[["game_pk", "home_team", "away_team"]]
            starters = starters.merge(games_lookup, on="game_pk", how="left")
            starters["opp_team"] = np.where(
                starters["is_home"] == True, starters["away_team"], starters["home_team"],
            )

        starters["own_bp_outs_2d"] = _prior_outs("team", "own", days=2)
        starters["opp_bp_outs_2d"] = _prior_outs("opp_team", "opp", days=2)
        n_own_missing = starters["own_bp_outs_2d"].isna().sum()
        n_opp_missing = starters["opp_bp_outs_2d"].isna().sum()
        if n_own_missing or n_opp_missing:
            print(f"  bullpen_daily lookups: own missing={n_own_missing}, opp missing={n_opp_missing}")
    else:
        print("  bullpen_daily empty - falling back to league avg for own_bp_outs_2d / opp_bp_outs_2d")
        starters["own_bp_outs_2d"] = np.nan
        starters["opp_bp_outs_2d"] = np.nan

    # Filter for games with exactly 2 teams
    starters = starters.groupby("game_pk").filter(lambda x: len(x) == 2)
    starters = starters.sort_values(["game_pk", "is_home"]).reset_index(drop=True)

    # Fill NaN features with league averages
    for col, fallback in LEAGUE_AVG.items():
        if col in starters.columns:
            n_missing = starters[col].isna().sum()
            if n_missing > 0:
                print(f"  Filling {n_missing} NaN values in '{col}' with league avg ({fallback})")
                starters[col] = starters[col].fillna(fallback)

    starters["starter_whip"] = starters.get("whip", starters.get("starter_whip", LEAGUE_AVG["whip"]))

    out_cols = [
        "game_pk", "game_date", "team", "pitcher_name", "is_home",
        "xfip", "starter_whip", "xfip_bullpen", "bullpen_k_9",
        "batting_ops", "batting_iso", "batting_k_pct",
        "park_factor", "actual_runs", "avg_last5", "avg_last10", "std_last5",
        "own_bp_outs_2d", "opp_bp_outs_2d",
    ]
    result = starters[[c for c in out_cols if c in starters.columns]].copy()
    result = result.rename(columns={"pitcher_name": "starter"})

    print(f"  Loaded {len(result)} rows ({result['game_pk'].nunique()} games)")
    return result


def preprocess_data(df):
    numeric_cols = [
        "xfip", "xfip_bullpen", "starter_whip", "bullpen_k_9",
        "batting_ops", "batting_iso", "batting_k_pct",
        "avg_last5", "avg_last10", "std_last5", "park_factor",
        "own_bp_outs_2d", "opp_bp_outs_2d",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
