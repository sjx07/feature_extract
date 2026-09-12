# Fresh libraries on the fork-join loop (runs/fresh, 2026-09-12)

Goal: rebuild the guidance libraries of text2cypher, science-quantitative, text2sql and math from fresh cold starts on the
new code (neighbourhoods, fork-join naming with folds and variants, plain-words prompts), keys from feature_extract/.env,
then the seed; budget $40; a granularity read at every step. Site for this workspace: port 8782 from the worktree.

## text2cypher v2 (job 8), settled by the round cap, $1.26
- 1,249 wordings; 126 features + 7 variants in 9 groups; reading coverage 92%; 57 wordings specific after a read.
- Cold start 57 features. Rounds added 69 features; rounds 3 to 8 added 1 to 4 each, so the pool was exhausted, not the cap.
- Largest features are the cold start's with real support (act as a query expert 43 prompts, translate natural language into
  a query 36, use only provided schema elements 36). Loop-born features read as single instructions (revise a query from
  error feedback, ensure nodes exist before creating relationships, convert short-form numeric values to plain numbers).
- Join: 1 proposal sent onto an existing feature, 1 duplicate folded; the narrower verdict produced 7 variants.
- One wrong-way variant: "act as a domain expert" (round 1) filed as a variant under the cold start's "act as a medical
  expert". The join can only say the younger is narrower; here the younger was broader. A "broader" verdict, making the
  older node the variant, would fix the orientation.
- No duplicate names. 16 indistinct pairs reported at the end, half of them the cold start's own (the three role features;
  code fence beside JSON beside "return output in a specified format"), left to a person by rule.
- 35 standing misfits, read as real disagreements, not rephrasings.
- The first fresh cold start of this corpus cited no wordings; the schema now requires a citation per feature.

## science-quantitative v1 (job 6), settled by the round cap, $1.03
- 993 wordings; 113 features + 9 variants in 11 groups (2 founded by the join); reading coverage 91%; 61 specific.
- Cold start 55 features; rounds 5 to 8 added 1 to 4 each.
- Largest: reason step by step 53 prompts, follow the specified format exactly 46 (with a 20-wording variant "use named
  answer schema fields"), act as a domain expert 42. Loop-born features single instructions (use placeholder for
  inapplicable fields, revise the solution after an error, consider sign and direction in grading).
- Join: 3 duplicates folded at birth; narrower verdicts made variants; 0 topics.
- Indistinct pairs left: the cold start's "return only the choice label / one number only / a binary label" trio and a
  grading cluster (compare with the reference answer, judge by the final result, mark answers as incorrect, penalize
  answer defects) reported in the last round, unruled because the run ended. The cold-start trio reads as three formats
  of one instruction; the rule that the cold start's pairs are never folded is worth revisiting.
- Two judge reasons came back in Chinese (DeepSeek); cosmetic.

## Over- and under-clustering, so far
- No bucket: the largest features are the cold start's and carry one instruction each; loop-born features sit at 3 to 16
  wordings with a median of about 6.
- Under-clustering is confined to pairs the loop is not allowed to fold (cold-start pairs) and to pairs reported in the
  final round. Both are visible on the page as reports.
