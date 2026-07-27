-- Basketball decision-support query examples.
-- These are intentionally simple and portable so the same logic can move into
-- a warehouse or internal analytics database.

-- Lineup summary used by basketball stakeholders.
SELECT
  lineup_id,
  lineup_profile,
  COUNT(*) AS possessions,
  AVG(actual_points) AS actual_ppp,
  AVG(predicted_points) AS predicted_ppp,
  AVG(defensive_risk) AS avg_defensive_risk,
  AVG(shot_quality) AS avg_shot_quality,
  AVG(turnover_pressure) AS avg_turnover_pressure,
  AVG(rebound_matchup) AS avg_rebound_matchup
FROM possession_predictions
GROUP BY lineup_id, lineup_profile
ORDER BY predicted_ppp DESC;

-- Tactical context audit.
SELECT
  tactic,
  COUNT(*) AS possessions,
  AVG(actual_points) AS actual_ppp,
  AVG(predicted_points) AS predicted_ppp,
  AVG(shot_quality) AS avg_shot_quality,
  AVG(turnover_pressure) AS avg_turnover_pressure
FROM possession_predictions
GROUP BY tactic
ORDER BY predicted_ppp DESC;

