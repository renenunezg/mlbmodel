BEGIN;

-- Which check allowed each prospect grade, so the site can say it accurately. The base pitcher model has not passed
-- its publish gate; tracked pitchers are graded through the stuff adjustment's own adoption rules instead.
ALTER TABLE mlb.prospect_values
  ADD COLUMN base_model_validated boolean,
  ADD COLUMN grade_gate text CHECK (grade_gate IN ('base_model', 'stuff_adjustment'));

COMMENT ON COLUMN mlb.prospect_values.base_model_validated IS
  'True when this player''s pool (hitters or pitchers) has a base model that passed its publish gate at fit time.';
COMMENT ON COLUMN mlb.prospect_values.grade_gate IS
  'The check that allowed this grade: base_model (the pool''s validated base model) or stuff_adjustment (pitchers: the stuff adjustment''s adoption rules). NULL when ungraded.';

NOTIFY pgrst, 'reload schema';

COMMIT;
