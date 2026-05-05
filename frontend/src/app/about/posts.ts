export type Post = {
  slug: string;
  date: string;       // "YYYY-MM-DD"
  title: string;
  summary: string;
  body: string;       // plain text or light markdown - rendered as paragraphs split on \n\n
};

// Add new entries at the top. Each entry appears as a card on the About page.
export const posts: Post[] = [
  {
    slug: "v2-coming",
    date: "2026-05-03",
    title: "V2 Is Coming and It's Bringing Bayes with It. Our Findings on XGBoost.",
    summary: "One month of live predictions with a 14-feature XGBoost regressor. What held up, what didn't, and why I'm going hierarchical Bayesian for v2.",
    body: "v1 came out of a simple bet: baseball is the highest-variance major sport, so don't overengineer it. One XGBoost regressor, 14 features (xFIP, bullpen workload, handedness-blended OPS, rolling scoring averages, park factor), expected runs as the output, with a fixed-dispersion negative binomial stacked on top to convert that into a win probability. I figured a smaller model would be harder to overfit on something this noisy.\n\nA month of live predictions in, MAE is sitting around 2.4 runs. That isn't bad in absolute terms, the game just resists prediction. The problem is what 2.4 runs means for totals: the line is asking for a distributional claim and a point estimate can't deliver one. The model spitting out 9.2 expected when the line is 8.5 sounds like edge until you remember the actual run distribution for that game probably runs from 5 to 13. I knew the totals issue going in and decided the moneyline output would carry the model. Mostly it hasn't.\n\nBrier was the worse number. Win-prob calibration came in slightly under coin-flip, not a collapse but not where I want to be. The pattern that bugs me most is comeback losses: the model has no representation of in-game leverage or pen state. I patched it mid-April with two bullpen-fatigue features (reliever outs over the prior 48 hours, per team) and they helped on the games where a tired pen blew a lead, but it's a patch on top of an output shape that was wrong to begin with. A scalar expected-runs number can't tell you that the tying run is on second in the seventh against a closer who threw 25 pitches yesterday.\n\nThe structural fix is to model the game as what it actually is, a sequence of plate appearances with compounding state, instead of a single number per team. v2 is a hierarchical Bayesian skill model fit with numpyro: batter outcome distributions by handedness, pitcher skill split by starter vs reliever, park effects priored on the values I already pull from Savant. The thing that sold me on going Bayesian here is partial pooling, which means a call-up with 40 PAs gets shrunk toward the league mean in proportion to how little I know about him, and I don't have to draw an arbitrary cutoff like 'you need 100 PAs to be in the model' which I'd otherwise have to defend.\n\nOn top of the skill layer there's a Monte Carlo simulator that runs each game 10,000 times against the posterior, inning by inning. The output isn't a number, it's a distribution over runs, which is the thing you actually need for totals and run line. Whether the calibration on those holds up under live betting is genuinely an open question, and Phase 6 of the rebuild is set up specifically to find out before any cutover happens. The Statcast data layer under all of this is unchanged from v1. The rethink is entirely in the modeling layer.",
  },
];
