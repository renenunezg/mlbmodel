import subprocess

print("🚀 Starting daily MLB model pipeline...")

scripts_to_run = [
    "backend/scrapers/scrape_team_batting_vs_lhp.py",
    "backend/scrapers/scrape_team_batting_vs_rhp.py",
    "backend/scrapers/scrape_starting_pitchers.py",
    "backend/scrapers/scrape_bullpen.py",
    "backend/scrapers/scrape_park_factors.py",
    "backend/scrapers/scrape_probable_starters.py",
    "backend/scrapers/scrape_runs_per_game.py",
    "backend/scrapers/scrape_season_scores.py",
    "backend/scrapers/scrape_odds.py",
    "backend/model.py",
    "backend/evaluate_model.py"
]

for script in scripts_to_run:
    print(f"\n▶️ Running {script}...")
    result = subprocess.run(["./env/bin/python", script], stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"❌ Script {script} failed with return code {result.returncode}")
        print("Error output:")
        print(result.stderr)
        break
    else:
        print(f"✅ Completed {script}.")

print("\n✅ Daily pipeline completed.")
