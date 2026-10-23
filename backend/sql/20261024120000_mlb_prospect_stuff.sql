BEGIN;

-- Per-pitch stuff for tracked prospect pitchers (Triple-A and Florida State League Statcast), null for hitters and for
-- pitchers under 200 tracked pitches in the scoring window. Packed to keep the table small:
-- pitch_mix: seven smallints per pitch type, ordered by usage: [type code, usage percent, velocity in tenths of mph,
--   induced vertical break and horizontal break in tenths of an inch (horizontal positive to the arm side), spin rpm,
--   20-80 stuff grade on the MLB scale]. Type codes: 1 FF, 2 SI, 3 FC, 4 SL, 5 ST, 6 SV, 7 CU, 8 KC, 9 CS, 10 CH, 11 FS,
--   12 FO, 13 SC (v2/prospects/stuff.py PITCH_CODES).
-- command_grade: 20-80 from location value beyond the count, null under the command sample floor.
-- stuff_level: 1 Triple-A, 2 Florida State League, 3 both; negative when the values could not be shifted to the MLB scale.
-- stuff_pitches: tracked pitches in the window.
ALTER TABLE mlb.prospect_values
  ADD COLUMN pitch_mix smallint[],
  ADD COLUMN command_grade smallint,
  ADD COLUMN stuff_level smallint,
  ADD COLUMN stuff_pitches smallint;

COMMIT;
