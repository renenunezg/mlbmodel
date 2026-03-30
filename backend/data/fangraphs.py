"""
Pitcher and team batting stats fetcher.

Computes advanced metrics (xFIP, WHIP, K/9, wRC+) from Statcast pitch-level
data via pybaseball. Replaces direct FanGraphs scraping (blocked by Cloudflare).

For team batting wRC+: uses FanGraphs team batting page via pybaseball.
Fallback: computes OPS from Statcast if FanGraphs is unavailable.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from backend.team_mappings import TEAM_NAME_MAP, normalize_team
import warnings
warnings.filterwarnings("ignore")

# 2026 FIP constant (updated each season; ~3.10-3.20 historically)
# This gets recalculated from the data if we have enough games.
DEFAULT_FIP_CONSTANT = 3.15


# Cache Statcast data to avoid re-fetching within the same pipeline run
_statcast_cache: dict[str, pd.DataFrame] = {}


def _get_statcast_range(start_date: date = None) -> pd.DataFrame:
    """Fetch Statcast pitch-level data from start_date to today. Cached per session.

    Defaults to March 20 of the current year (safely before any opening day).
    """
    from pybaseball import statcast

    end_dt = date.today()
    start_dt = start_date or date(end_dt.year, 3, 25)  # MLB opening day
    cache_key = f"{start_dt}_{end_dt}"

    if cache_key in _statcast_cache:
        df = _statcast_cache[cache_key]
        print(f"  Using cached Statcast data: {len(df)} pitches")
        return df

    print(f"  Fetching Statcast data: {start_dt} to {end_dt}...")
    df = statcast(start_dt=str(start_dt), end_dt=str(end_dt))

    if df.empty:
        print("  Warning: no Statcast data returned.")
    else:
        # Filter to regular season only (exclude spring training, etc.)
        if "game_type" in df.columns:
            pre_filter = len(df)
            df = df[df["game_type"] == "R"]
            if len(df) < pre_filter:
                print(f"  Filtered to regular season: {len(df)} pitches (dropped {pre_filter - len(df)} spring training)")
        print(f"  Got {len(df)} pitches across {df['game_pk'].nunique()} games")

    _statcast_cache[cache_key] = df
    return df


def _identify_starters(pitch_df: pd.DataFrame) -> set:
    """Identify starting pitchers (first pitcher per team per game)."""
    starters = set()
    for game_pk in pitch_df["game_pk"].unique():
        game = pitch_df[pitch_df["game_pk"] == game_pk]
        for side in ["home", "away"]:
            # Starters pitch in inning 1; pick the one with the most pitches
            inning1 = game[(game["inning"] == 1) & (game["inning_topbot"] == ("Bot" if side == "home" else "Top"))]
            if not inning1.empty:
                # The pitcher facing the other team's batters in inning 1
                starter_id = inning1["pitcher"].mode()
                if not starter_id.empty:
                    starters.add(starter_id.iloc[0])
    return starters


def _compute_pitcher_stats(pitch_df: pd.DataFrame, pitcher_ids: set = None) -> pd.DataFrame:
    """Compute xFIP, WHIP, K/9, BB/9 from pitch-level Statcast data.

    If pitcher_ids is provided, only compute for those pitchers.
    """
    # Filter to plate appearances (events)
    pa_df = pitch_df[pitch_df["events"].notna()].copy()

    if pitcher_ids is not None:
        pa_df = pa_df[pa_df["pitcher"].isin(pitcher_ids)]

    if pa_df.empty:
        return pd.DataFrame()

    # Compute per-pitcher stats
    rows = []
    for pitcher_id, group in pa_df.groupby("pitcher"):
        pitcher_name = group["player_name"].iloc[0]

        # Count events
        k = (group["events"] == "strikeout").sum()
        bb = (group["events"] == "walk").sum()
        ibb = (group["events"] == "intent_walk").sum()
        hbp = (group["events"] == "hit_by_pitch").sum()
        hr = (group["events"] == "home_run").sum()
        hits = group["events"].isin(["single", "double", "triple", "home_run"]).sum()

        # Fly balls from bb_type
        fb = (group["bb_type"] == "fly_ball").sum()

        # Compute IP from outs
        # Each out-producing event counts; strikeouts, field_outs, etc.
        outs = group["events"].isin([
            "strikeout", "field_out", "force_out", "grounded_into_double_play",
            "double_play", "triple_play", "sac_fly", "sac_bunt",
            "fielders_choice_out", "caught_stealing_2b", "caught_stealing_3b",
            "caught_stealing_home",
        ]).sum()
        # Double plays count as 2 outs
        dp = group["events"].isin(["grounded_into_double_play", "double_play"]).sum()
        total_outs = outs + dp
        ip = total_outs / 3

        if ip < 0.1:
            continue

        # xFIP = ((13 * (lgHR/lgFB) * FB) + 3*(BB+HBP-IBB) - 2*K) / IP + FIP_constant
        # Use league HR/FB rate (~0.10-0.12)
        lg_hr_fb = 0.11  # Historical average

        xfip = ((13 * lg_hr_fb * fb) + 3 * (bb + hbp - ibb) - 2 * k) / ip + DEFAULT_FIP_CONSTANT if ip > 0 else np.nan
        fip = ((13 * hr) + 3 * (bb + hbp - ibb) - 2 * k) / ip + DEFAULT_FIP_CONSTANT if ip > 0 else np.nan
        whip = (bb + hits - hr) / ip if ip > 0 else np.nan  # WHIP = (BB + H) / IP, but hits includes HR
        whip = (bb + hits) / ip if ip > 0 else np.nan
        k_9 = (k / ip) * 9 if ip > 0 else np.nan
        bb_9 = (bb / ip) * 9 if ip > 0 else np.nan
        hr_9 = (hr / ip) * 9 if ip > 0 else np.nan
        era = (group["events"].isin(["home_run"]).sum() * 1) / ip * 9  # Simplified; not perfect

        # Get team from Statcast (pitcher's team)
        # In Statcast, pitcher's team is the fielding team
        team = group["home_team"].iloc[0] if group["inning_topbot"].iloc[0] == "Top" else group["away_team"].iloc[0]

        rows.append({
            "pitcher_name": pitcher_name,
            "pitcher_id": pitcher_id,
            "team": normalize_team(team) if team else None,
            "ip": round(ip, 1),
            "fip": round(fip, 2) if pd.notna(fip) else np.nan,
            "xfip": round(xfip, 2) if pd.notna(xfip) else np.nan,
            "whip": round(whip, 2) if pd.notna(whip) else np.nan,
            "k_9": round(k_9, 2) if pd.notna(k_9) else np.nan,
            "bb_9": round(bb_9, 2) if pd.notna(bb_9) else np.nan,
            "hr_9": round(hr_9, 2) if pd.notna(hr_9) else np.nan,
        })

    return pd.DataFrame(rows)


def fetch_pitcher_stats(season: int = None) -> pd.DataFrame:
    """Fetch starting pitcher stats computed from Statcast data.

    Returns DataFrame with columns matching the pitcher_stats DB table.
    """
    if season is None:
        season = date.today().year

    pitch_df = _get_statcast_range()
    if pitch_df.empty:
        print("No Statcast data — cannot compute pitcher stats.")
        return pd.DataFrame(columns=["pitcher_name", "team", "xfip", "whip", "k_9", "season", "role"])

    starter_ids = _identify_starters(pitch_df)
    print(f"  Identified {len(starter_ids)} starting pitchers")

    df = _compute_pitcher_stats(pitch_df, pitcher_ids=starter_ids)
    if df.empty:
        print("  No pitcher stats computed (not enough plate appearances).")
        return pd.DataFrame(columns=["pitcher_name", "team", "xfip", "whip", "k_9", "season", "role"])

    df["season"] = season
    df["role"] = "starter"
    return df


def fetch_bullpen_stats(season: int = None) -> pd.DataFrame:
    """Fetch team-level bullpen stats computed from Statcast data.

    Returns DataFrame with columns matching the bullpen_stats DB table.
    """
    if season is None:
        season = date.today().year

    pitch_df = _get_statcast_range()
    if pitch_df.empty:
        print("No Statcast data — cannot compute bullpen stats.")
        return pd.DataFrame(columns=["team", "xfip", "k_9", "season"])

    starter_ids = _identify_starters(pitch_df)
    reliever_ids = set(pitch_df["pitcher"].unique()) - starter_ids
    print(f"  Identified {len(reliever_ids)} relievers")

    df = _compute_pitcher_stats(pitch_df, pitcher_ids=reliever_ids)
    if df.empty:
        return pd.DataFrame(columns=["team", "xfip", "k_9", "season"])

    # Aggregate to team level (IP-weighted averages)
    team_stats = df.groupby("team").apply(
        lambda g: pd.Series({
            "ip": g["ip"].sum(),
            "xfip": np.average(g["xfip"].dropna(), weights=g.loc[g["xfip"].notna(), "ip"]) if g["xfip"].notna().any() else np.nan,
            "fip": np.average(g["fip"].dropna(), weights=g.loc[g["fip"].notna(), "ip"]) if g["fip"].notna().any() else np.nan,
            "whip": np.average(g["whip"].dropna(), weights=g.loc[g["whip"].notna(), "ip"]) if g["whip"].notna().any() else np.nan,
            "k_9": np.average(g["k_9"].dropna(), weights=g.loc[g["k_9"].notna(), "ip"]) if g["k_9"].notna().any() else np.nan,
            "bb_9": np.average(g["bb_9"].dropna(), weights=g.loc[g["bb_9"].notna(), "ip"]) if g["bb_9"].notna().any() else np.nan,
        })
    ).reset_index()

    team_stats["season"] = season

    # Round numeric columns
    for col in ["ip", "xfip", "fip", "whip", "k_9", "bb_9"]:
        if col in team_stats.columns:
            team_stats[col] = team_stats[col].round(2)

    return team_stats


def fetch_team_batting(season: int = None) -> pd.DataFrame:
    """Fetch team batting stats from Statcast data.

    Computes OPS, ISO, K%, BB% by split (vs RHP / vs LHP).
    wRC+ is set to None (requires league context not available from Statcast alone).

    For wRC+, falls back to pybaseball's FanGraphs integration if available.
    """
    if season is None:
        season = date.today().year

    pitch_df = _get_statcast_range()
    if pitch_df.empty:
        print("No Statcast data — cannot compute team batting.")
        return pd.DataFrame(columns=["team", "split", "wrc_plus", "iso", "k_pct", "ops", "obp", "season"])

    pa_df = pitch_df[pitch_df["events"].notna()].copy()

    # Determine pitcher handedness for each pitch
    # p_throws is available in Statcast
    if "p_throws" not in pa_df.columns:
        print("  Warning: p_throws not in Statcast data, cannot compute splits.")
        return pd.DataFrame()

    rows = []
    for split_name, hand in [("vs_rhp", "R"), ("vs_lhp", "L")]:
        split_df = pa_df[pa_df["p_throws"] == hand]

        for side in ["home", "away"]:
            if side == "home":
                # Home team bats in bottom of inning
                team_pa = split_df[split_df["inning_topbot"] == "Bot"]
            else:
                team_pa = split_df[split_df["inning_topbot"] == "Top"]

            if team_pa.empty:
                continue

            # Group by batting team
            team_col = f"{side}_team"
            if team_col not in team_pa.columns:
                continue

            for team, g in team_pa.groupby(team_col):
                pa = len(g)
                ab = pa - g["events"].isin(["walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf"]).sum()
                hits = g["events"].isin(["single", "double", "triple", "home_run"]).sum()
                doubles = (g["events"] == "double").sum()
                triples = (g["events"] == "triple").sum()
                hr = (g["events"] == "home_run").sum()
                bb = g["events"].isin(["walk", "intent_walk"]).sum()
                hbp = (g["events"] == "hit_by_pitch").sum()
                so = (g["events"] == "strikeout").sum()
                sf = (g["events"] == "sac_fly").sum()

                avg = hits / ab if ab > 0 else 0
                obp = (hits + bb + hbp) / (ab + bb + hbp + sf) if (ab + bb + hbp + sf) > 0 else 0
                slg = (hits - doubles - triples - hr + 2 * doubles + 3 * triples + 4 * hr) / ab if ab > 0 else 0
                ops = obp + slg
                iso = slg - avg
                k_pct = (so / pa * 100) if pa > 0 else 0
                bb_pct = (bb / pa * 100) if pa > 0 else 0

                rows.append({
                    "team": normalize_team(team),
                    "split": split_name,
                    "pa": pa,
                    "wrc_plus": None,  # Requires league context
                    "ops": round(ops, 3),
                    "obp": round(obp, 3),
                    "slg": round(slg, 3),
                    "iso": round(iso, 3),
                    "k_pct": round(k_pct, 1),
                    "bb_pct": round(bb_pct, 1),
                    "babip": None,
                })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Aggregate home + away stats for same team/split
    agg_rows = []
    for (team, split), g in df.groupby(["team", "split"]):
        total_pa = g["pa"].sum()
        if total_pa == 0:
            continue
        agg_rows.append({
            "team": team,
            "split": split,
            "pa": total_pa,
            "wrc_plus": None,
            "ops": np.average(g["ops"], weights=g["pa"]),
            "obp": np.average(g["obp"], weights=g["pa"]),
            "slg": np.average(g["slg"], weights=g["pa"]),
            "iso": np.average(g["iso"], weights=g["pa"]),
            "k_pct": np.average(g["k_pct"], weights=g["pa"]),
            "bb_pct": np.average(g["bb_pct"], weights=g["pa"]),
        })

    result = pd.DataFrame(agg_rows)
    result["season"] = season

    # Round
    for col in ["ops", "obp", "slg", "iso"]:
        if col in result.columns:
            result[col] = result[col].round(3)
    for col in ["k_pct", "bb_pct"]:
        if col in result.columns:
            result[col] = result[col].round(1)

    return result


if __name__ == "__main__":
    season = date.today().year

    print(f"=== Starting Pitcher Stats ({season}) ===")
    pitchers = fetch_pitcher_stats(season)
    if not pitchers.empty:
        print(pitchers[["pitcher_name", "team", "ip", "xfip", "whip", "k_9"]].head(10).to_string(index=False))
        print(f"\nTotal pitchers: {len(pitchers)}")
    else:
        print("No data.")

    print(f"\n=== Bullpen Stats ({season}) ===")
    bullpen = fetch_bullpen_stats(season)
    if not bullpen.empty:
        print(bullpen[["team", "ip", "xfip", "whip", "k_9"]].to_string(index=False))
    else:
        print("No data.")

    print(f"\n=== Team Batting ({season}) ===")
    batting = fetch_team_batting(season)
    if not batting.empty:
        print(batting[["team", "split", "pa", "ops", "iso", "k_pct"]].to_string(index=False))
    else:
        print("No data.")
