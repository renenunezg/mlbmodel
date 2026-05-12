# Bullpen Rest Predicts MLB Game Outcomes Better Than We Expected

*Draft - pulled from `analysis/outputs/03_bullpen_fatigue/` and #5 of the April 30 analysis session.*

---

## The tweet (~280 chars)

> Looked at 421 MLB games to see what predicts our model's wins. Hottest signal: opposing team's bullpen IP in the prior 2 days. Picks against rested bullpens: -5.6% ROI. Picks against gassed bullpens: +13.3% ROI. The book isn't fully pricing in pen fatigue. Thread 👇

## The thread (5-6 tweets)

**1/** I run an MLB run-prediction model. Recent stretch of comeback losses got me asking: how much of the model's pain is bullpen volatility? Pulled 421 completed games and dug in.

**2/** Started with the obvious question: of the games where my pick lost, how often was the pick winning through inning 5? Just **20.7%**. Most losses (62%) were already in the bag by the 5th. Bullpen meltdowns aren't the main story.

**3/** But when I broke losses down by model confidence, something interesting popped: high-confidence losses (75%+) are almost never comebacks. They're blowouts where the model was wrong from pitch 1. **The model has a feature gap, not a bullpen problem.**

**4/** So I built a feature: "reliever outs the opposing team has thrown in the prior 2 days." Bucketed picks into low / mid / high opp-bullpen-fatigue.

**5/**
- Low fatigue (rested pen): 55.5% win rate, **−5.57% ROI**
- Mid: 50.4%, −9.61%
- High fatigue (gassed pen): 58.3%, **+13.32% ROI**

139 games at high fatigue. Wide CI, but the directional story holds across 1d/2d/3d windows. The 1-day window is even sharper: high-fatigue opp = 64.0% win rate.

**6/** Even cleaner: my pick's *own* bullpen rest matters too. Picks with a rested own-bullpen win 60.2% vs 52.4% when fatigued - a **~8-percentage-point swing**. Both features now live in the model (it's 14 features as of this rev, up from 12). Will report back with backtest results in a couple weeks.

---

## The full blog post (1,200 words)

### Hook

Anyone who's bet MLB long enough has had this thought: I should just bet the first 5 innings. The starter is what I have a read on. The bullpen is chaos. Half the sportsbook regulars I've talked to swear by F5 markets for exactly this reason - strip the bullpen variance out and the model has a cleaner signal.

I wanted to test the premise. So I pulled 421 completed games from this season's run of my expected-runs model, fetched line scores from the MLB Stats API, and asked a deceptively simple question: **how many of the model's losses are actually bullpen-driven comebacks?**

The answer surprised me, and led somewhere more interesting.

### What "bullpen volatility" actually means in the data

For every game where my model's pick lost, I computed the score through the end of inning 5 - the F5 cutoff. If my pick was leading after 5 and lost the game, that's a *comeback loss*: the bullpen blew it. If my pick was already trailing or tied through 5, the bullpen wasn't the problem.

Here's the breakdown across 184 model losses:

| State through inning 5 | Losses | Share |
|---|---|---|
| Pick was leading | 38 | **20.7%** |
| Pick was tied | 32 | 17.4% |
| Pick was trailing | 114 | 62.0% |

Only **1 in 5 losses** are true bullpen comebacks. The majority - 62% - are games where the model was already losing through 5. Those aren't pen meltdowns. Those are bad picks.

The F5-only crowd is solving a problem that explains a fifth of the pain.

### Where it gets interesting: confidence

When I sliced losses by model confidence, something jumped out:

| Confidence | Losses | Comeback rate |
|---|---|---|
| 50-55% | 57 | 17.5% |
| 55-65% | 65 | 26.2% |
| 65-75% | 32 | 28.1% |
| **75%+** | **30** | **6.7%** |

The very-high-confidence losses are *not* comebacks. They're blowouts. When the model says "75%+ confidence" and loses, in 93% of cases the pick was already losing through 5.

That tells me something specific: **the model isn't getting beaten by bullpen variance at the high end. It's getting beaten by feature gaps.** When it confidently picks the wrong team, it's missing some pre-game signal - and bullpen volatility, the convenient narrative, is taking the blame for what's actually a model defect.

### The feature I should have built earlier

If the model has a feature gap, what's missing? I went looking, and the obvious candidate was bullpen state - not "is the bullpen good in aggregate" (we already have that as `xfip_bullpen`), but "is the bullpen rested today, specifically."

I pulled boxscores for every completed game and computed, per team per game, **how many reliever outs that team had thrown in the prior 2 days**. Then I bucketed my model's picks into low / mid / high *opponent* bullpen fatigue - the "their pen is gassed and I'm betting against them" axis.

| Opp BP fatigue | n | Win rate | ROI on flat 1u |
|---|---|---|---|
| Low (well-rested) | 155 | 55.5% | −5.57% |
| Mid | 127 | 50.4% | −9.61% |
| **High (fatigued)** | **139** | **58.3%** | **+13.32%** |

The picks against fatigued bullpens are profitable. Not at borderline edges - at +13%. The book isn't fully pricing in opp bullpen exhaustion, and the model's existing features happen to already correlate with picks that exploit it.

(Quick methodology note that mattered: my first pass counted any pitcher who wasn't the listed starter as a "reliever." That over-counted opener-strategy games - where a 1-IP "starter" precedes a 5-IP bulk reliever - by attributing the bulk reliever's outs to the bullpen tally. After re-tagging any first pitcher with fewer than 3 IP as a functional reliever, **7.1% of team-games** flipped classification. Real number of openers in modern MLB is bigger than I thought.)

The flip side, **own bullpen rest**, is even cleaner:

| Own BP fatigue | n | Win rate |
|---|---|---|
| Low (rested) | 166 | **60.2%** |
| Mid | 129 | 50.4% |
| High (fatigued) | 126 | 52.4% |

Nearly an 8-point swing. When the model's pick has a rested bullpen behind them, they win 60.2% of the time. When they don't, 52.4%. The model - at the time of this analysis - knew nothing about this.

### Caveats before I get yelled at

- **Sample size**: 413 games is one month of data. The 95% CI on a 124-game bucket's win rate is roughly ±5 points. The directional story is consistent across 1d / 2d / 3d windows, which gives me more confidence than a single number, but I'd want another month before calling these effects validated.
- **Reliever-outs counting has noise.** Openers (a 1-IP "starter" followed by a bulk reliever) inflate reliever-outs for those teams. ~3% of games. Doesn't change the direction.
- **Causation vs. correlation**: a fatigued bullpen often correlates with the team being on a long road trip, playing in extra innings recently, etc. The feature might be picking up "team is in a rough patch" more than "the relievers' arms are tired." Both are predictive; the framing matters less for a model than for a narrative.

### What I did about it

Added two features to the model:

- `own_bp_outs_2d` - reliever outs by my pick's team in prior 2 days
- `opp_bp_outs_2d` - reliever outs by the opposing team in prior 2 days

12 features → 14. Retraining the XGBoost regressor on the same training set put `opp_bp_outs_2d` at importance rank 8/14 (above bullpen K/9, std_last5, and park_factor) and `own_bp_outs_2d` at 12/14 (above avg_last10). Both new features earned their slot. CV MAE was 2.674 on 700 OOF folds - directionally similar to the 12-feature model; the proper test is forward-looking ROI over the next month, which I'll post separately.

### What this taught me about model interpretation

The thing I keep coming back to: **the bullpen-comeback narrative was satisfying because it explained variance without challenging the model.** If losses are bullpen volatility, the model is fine, the world is just noisy. Comforting.

The data said: 1 in 5 of your losses fit that story. The other 4 in 5 are you being wrong about something pre-game. Specifically, the high-confidence losses - the ones that hurt the most - are you being *very wrong* about something pre-game.

That's a less comforting answer. It's also the answer that points to actual improvements.

---

*Code, charts, and CSVs for everything above live in `analysis/outputs/03_bullpen_fatigue/`. Anyone running a similar model: I'd love to see your version of the comeback-rate-by-confidence table.*
