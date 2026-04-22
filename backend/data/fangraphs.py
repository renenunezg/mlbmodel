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
from pathlib import Path
from backend.team_mappings import normalize_team
import warnings
warnings.filterwarnings("ignore")

# FIP constant — league-wide reconciliation factor (~3.10-3.20 historically).
# Updated each season.
DEFAULT_FIP_CONSTANT = 3.15

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache"

# Cache Statcast data to avoid re-fetching within the same pipeline run
_statcast_cache: dict[str, pd.DataFrame] = {}


def _get_statcast_range(start_date: date = None) -> pd.DataFrame:
    """Fetch Statcast pitch-level data from start_date to today. Cached per session.

    Defaults to March 25 of the current year (safely before any opening day).
    """
    from pybaseball import statcast

    end_dt = date.today()
    start_dt = start_date or date(end_dt.year, 3, 25)
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


def _compute_pitcher_stats(pitch_df: pd.DataFrame, pitcher_ids: set = None,
                            starter_ids: set = None) -> pd.DataFrame:
    """Compute xFIP, WHIP, K/9, BB/9 from pitch-level Statcast data.

    If pitcher_ids is provided, only compute for those pitchers.
    If starter_ids is provided, also computes avg_ip_per_start (IP per game started)
    for pitchers in that set.
    """
    # Filter to plate appearances (events)
    pa_df = pitch_df[pitch_df["events"].notna()].copy()

    if pitcher_ids is not None:
        pa_df = pa_df[pa_df["pitcher"].isin(pitcher_ids)]

    if pa_df.empty:
        return pd.DataFrame()

    # Pre-compute IP per (pitcher, game) so we can derive avg IP per start for starters.
    # This is also what the 60/40 blend replacement uses to weight starter vs bullpen innings.
    per_game_outs = None
    if starter_ids is not None and pa_df["pitcher"].isin(starter_ids).any():
        out_events = {
            "strikeout", "field_out", "force_out", "grounded_into_double_play",
            "double_play", "triple_play", "sac_fly", "sac_bunt",
            "fielders_choice_out", "caught_stealing_2b", "caught_stealing_3b",
            "caught_stealing_home",
        }
        dp_events = {"grounded_into_double_play", "double_play"}
        pa_df["_outs"] = pa_df["events"].isin(out_events).astype(int) + pa_df["events"].isin(dp_events).astype(int)
        per_game_outs = pa_df.groupby(["pitcher", "game_pk"])["_outs"].sum()

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
        whip = (bb + hits) / ip if ip > 0 else np.nan
        k_9 = (k / ip) * 9 if ip > 0 else np.nan
        bb_9 = (bb / ip) * 9 if ip > 0 else np.nan
        hr_9 = (hr / ip) * 9 if ip > 0 else np.nan

        # Pitcher's team is the fielding team (top of inning = home pitching).
        team = group["home_team"].iloc[0] if group["inning_topbot"].iloc[0] == "Top" else group["away_team"].iloc[0]

        # Pitcher throwing hand (for bullpen RHP-share aggregation).
        p_throws = group["p_throws"].iloc[0] if "p_throws" in group.columns and not group["p_throws"].empty else None

        # Avg IP per start — only meaningful for starters. Counts games where this
        # pitcher recorded at least 6 outs (2 IP) to exclude bullpen appearances.
        avg_ip_per_start = np.nan
        if per_game_outs is not None and pitcher_id in starter_ids:
            pg = per_game_outs.loc[pitcher_id]
            pg = pg[pg >= 6]  # real starts only
            if len(pg) > 0:
                avg_ip_per_start = round(float(pg.mean()) / 3, 2)

        rows.append({
            "pitcher_name": pitcher_name,
            "pitcher_id": pitcher_id,
            "team": normalize_team(team) if team else None,
            "p_throws": p_throws,
            "ip": round(ip, 1),
            "fip": round(fip, 2) if pd.notna(fip) else np.nan,
            "xfip": round(xfip, 2) if pd.notna(xfip) else np.nan,
            "whip": round(whip, 2) if pd.notna(whip) else np.nan,
            "k_9": round(k_9, 2) if pd.notna(k_9) else np.nan,
            "bb_9": round(bb_9, 2) if pd.notna(bb_9) else np.nan,
            "hr_9": round(hr_9, 2) if pd.notna(hr_9) else np.nan,
            "avg_ip_per_start": avg_ip_per_start,
        })

    return pd.DataFrame(rows)


def _get_prior_season_pitcher_stats() -> pd.DataFrame:
    """Get prior-season pitcher stats, cached to parquet for fast reuse.

    Used as fallback for pitchers with no current-season data.
    """
    prior_year = date.today().year - 1
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"pitcher_stats_{prior_year}.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  Using cached {prior_year} pitcher stats: {len(df)} pitchers")
        return df

    print(f"  Fetching {prior_year} Statcast data for pitcher fallback (one-time)...")
    from pybaseball import statcast

    # Fetch full prior season in monthly chunks to avoid timeouts
    all_chunks = []
    for month in range(3, 11):  # March through October
        start = date(prior_year, month, 1)
        if month == 10:
            end = date(prior_year, 10, 31)
        else:
            end = date(prior_year, month + 1, 1) - timedelta(days=1)
        try:
            chunk = statcast(start_dt=str(start), end_dt=str(end))
            if not chunk.empty:
                all_chunks.append(chunk)
                print(f"    {start.strftime('%b')}: {len(chunk)} pitches")
        except Exception as e:
            print(f"    {start.strftime('%b')}: failed ({e})")

    if not all_chunks:
        print(f"  No {prior_year} Statcast data available")
        return pd.DataFrame()

    pitch_df = pd.concat(all_chunks, ignore_index=True)
    if "game_type" in pitch_df.columns:
        pitch_df = pitch_df[pitch_df["game_type"] == "R"]

    # Compute stats for all pitchers (not just starters — they might start next year)
    df = _compute_pitcher_stats(pitch_df)
    if df.empty:
        return pd.DataFrame()

    df["season"] = prior_year
    df.to_parquet(cache_path, index=False)
    print(f"  Cached {len(df)} pitcher stats to {cache_path}")
    return df


def fetch_pitcher_stats(season: int = None) -> pd.DataFrame:
    """Fetch starting pitcher stats computed from Statcast data.

    For pitchers with no current-season stats, falls back to prior-season data.
    Returns DataFrame with columns matching the pitcher_stats DB table.
    """
    if season is None:
        season = date.today().year

    pitch_df = _get_statcast_range()

    current_stats = pd.DataFrame()
    if not pitch_df.empty:
        starter_ids = _identify_starters(pitch_df)
        print(f"  Identified {len(starter_ids)} starting pitchers")
        current_stats = _compute_pitcher_stats(
            pitch_df, pitcher_ids=starter_ids, starter_ids=starter_ids,
        )

    # Get prior-season stats as fallback
    prior_stats = _get_prior_season_pitcher_stats()

    if current_stats.empty and prior_stats.empty:
        print("  No pitcher stats from current or prior season.")
        return pd.DataFrame(columns=["pitcher_name", "team", "xfip", "whip", "k_9", "season", "role"])

    if current_stats.empty:
        # All fallback
        print(f"  Using {len(prior_stats)} prior-season pitcher stats as fallback")
        prior_stats["season"] = season
        prior_stats["role"] = "starter"
        prior_stats = prior_stats.drop(columns=["p_throws"], errors="ignore")
        return prior_stats

    current_stats["season"] = season
    current_stats["role"] = "starter"

    # p_throws is only needed internally (for bullpen RHP-share aggregation);
    # the pitcher_stats DB table has no such column.
    current_stats = current_stats.drop(columns=["p_throws"], errors="ignore")

    if prior_stats.empty:
        return current_stats

    # Merge: current season takes priority, prior season fills gaps by pitcher_id
    current_ids = set(current_stats["pitcher_id"].dropna().unique())
    prior_fallback = prior_stats[~prior_stats["pitcher_id"].isin(current_ids)].copy()

    if not prior_fallback.empty:
        prior_fallback["season"] = season
        prior_fallback["role"] = "starter"
        prior_fallback = prior_fallback.drop(columns=["p_throws"], errors="ignore")
        print(f"  Added {len(prior_fallback)} prior-season pitcher stats as fallback")
        return pd.concat([current_stats, prior_fallback], ignore_index=True)

    return current_stats


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
    def _ip_weighted(g, col):
        mask = g[col].notna()
        if not mask.any():
            return np.nan
        return np.average(g.loc[mask, col], weights=g.loc[mask, "ip"])

    def _rhp_ip_share(g):
        total_ip = g["ip"].sum()
        if total_ip <= 0:
            return np.nan
        rhp_ip = g.loc[g["p_throws"] == "R", "ip"].sum()
        return rhp_ip / total_ip

    team_stats = df.groupby("team").apply(
        lambda g: pd.Series({
            "ip": g["ip"].sum(),
            "xfip": _ip_weighted(g, "xfip"),
            "fip": _ip_weighted(g, "fip"),
            "whip": _ip_weighted(g, "whip"),
            "k_9": _ip_weighted(g, "k_9"),
            "bb_9": _ip_weighted(g, "bb_9"),
            "rhp_ip_share": _rhp_ip_share(g),
        })
    ).reset_index()

    team_stats["season"] = season

    # Round numeric columns
    for col in ["ip", "xfip", "fip", "whip", "k_9", "bb_9"]:
        if col in team_stats.columns:
            team_stats[col] = team_stats[col].round(2)
    team_stats["rhp_ip_share"] = team_stats["rhp_ip_share"].round(3)

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
