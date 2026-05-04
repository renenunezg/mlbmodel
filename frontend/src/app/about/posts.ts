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
    body: "v1 started from a reasonable prior: baseball is the highest-variance major sport, so probably don't overthink it. One XGBoost regressor, 14 features, expected runs as output, negative binomial on top for win probabilities. The features were real (xFIP, bullpen workload, handedness-blended OPS, rolling scoring averages, park factor) and the architecture was intentionally lean. My working hypothesis was that complexity backfires in high-variance domains. You're mostly just fitting the noise in fancier ways.\n\nOne month in, the MAE is sitting around 2.4 runs. For a baseball model that's not shameful. The game resists prediction. But 2.4 runs is more than enough to get consistently wrecked on totals, because a totals line is asking you to make a distributional claim and a point estimate can't give you that. The model saying 9.2 when the line is 8.5 is not a useful edge if the actual run distribution spans from 5 to 13. I knew this going in and convinced myself it was fine. It's not fine.\n\nThe Brier score was the harder number. Win probability calibration came in slightly below coin-flip. Not a collapse, but not good. Comeback losses were a consistent pattern: the model had no representation of in-game leverage or pen state. I added two bullpen fatigue features mid-April (reliever outs over the prior 48 hours for each team). They helped on games where an overworked pen gave up a late lead. But it was a patch. A single expected-runs output is structurally wrong for a game that's a sequence of plate appearances with compounding state. It doesn't know the tying run is on second in the seventh with a tired closer.\n\nThe structural answer is to model the game as a sequence. v2 is a hierarchical Bayesian skill model fit with numpyro: batter wOBA by handedness, pitcher FIP by role (starter vs. reliever), park effects with priors from the existing park factors table. Bayesian inference is a framework I like a lot, and partial pooling is the thing that makes it worth the setup cost here. A call-up with 40 PAs gets shrunk toward the mean in proportion to how little data you have on him. That's exactly the right behavior for a problem where the gap between a 40-PA sample and a 600-PA sample is enormous, and there's no clean threshold to draw instead.\n\nOn top of that, a Monte Carlo simulator runs each game 50,000 times through the posterior distributions. The output isn't a run expectancy number, it's a distribution over runs. Which is what you need to actually say something useful about a totals line. The Statcast data layer under all of this is unchanged from v1. The rethink is entirely in the modeling layer.",
  },
];
