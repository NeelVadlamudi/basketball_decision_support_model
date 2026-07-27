# Basketball Decision Support Model

This is a public, reproducible basketball analytics work sample built for roles that ask for decision-support models, validation, and clear communication to basketball stakeholders.

The project uses synthetic possession-level and player-role data shaped like basketball operations work. It does not use internal data, scouting grades, tracking data, or proprietary team context. The point is to demonstrate the workflow: turn an ambiguous basketball question into a model, validate it, communicate uncertainty, and produce a decision-ready output.

## Basketball Question

Which player groups and lineup profiles create the best decision tradeoff once offensive efficiency, defensive risk, sample size, uncertainty, role fit, and tactical context are considered together?

## What This Builds

- Possession-level synthetic dataset across lineups, opponents, game context, and tactical features
- SQL aggregation layer for lineup and player decision tables
- Ridge model for possession value prediction
- Backtest split by game window to reduce leakage
- Uncertainty bands using bootstrap resampling
- Lineup decision board with practical basketball notes
- Player role fit matrix for spacing, creation, defensive versatility, rebounding, and ball security
- Executive brief written for technical and non-technical readers

## Why This Fits Basketball Analytics

The role asks for someone who can evaluate players, teams, lineups, tactics, and basketball decision-making questions. This project demonstrates that structure in a small, reviewable package:

- translates a vague basketball question into a measurable analytical plan
- treats uncertainty and sample size as part of the recommendation
- keeps model outputs interpretable for non-technical stakeholders
- documents assumptions, limitations, and validation checks
- creates outputs that Engineering could productionize later

## Run

```bash
python3 basketball_decision_model.py
```

Generated outputs are written to `outputs/`.

## Main Outputs

- `outputs/lineup_decision_board.csv`
- `outputs/player_role_fit.csv`
- `outputs/model_validation.md`
- `outputs/executive_brief.md`
- `outputs/lineup_net_rating_uncertainty.png`
- `outputs/player_role_fit_matrix.png`
- `outputs/tactical_feature_importance.png`

## Result Snapshot

The current run creates 4,428 synthetic possessions and a test-period decision board across 12 lineup profiles.

- Validation MAE: 0.269 points per possession
- Validation RMSE: 0.340 points per possession
- Top aggregate tests: balanced playoff-style unit, spacing with two-way wings
- Main decision guardrail: single-possession R-squared is modest by design, so recommendations are made from aggregate ranking, uncertainty bands, sample size, and basketball context notes

## Notes

Because public applicants do not have access to team-only tracking, scouting, and internal player context, the dataset is synthetic. In a team environment, the same workflow would connect to internal tracking/spatiotemporal data, play-by-play, lineup/personnel data, scouting notes, and player development context.
