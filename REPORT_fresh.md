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

## text2sql v1 (job 11), settled by the round cap, $4.63
- 5,639 wordings; 385 features + 38 variants in 16 groups (8 founded by the join); reading coverage 96%; 151 specific.
- Cold start 64 features over 4,695 of the wordings (944 waited for the loop). Round 1 named 242 features and 22 variants,
  folded 24 duplicates at birth; rounds 5 to 8 added 3 to 9 each.
- Join over the run: 133 judge pairs ruled, 27 folded, 21 narrowed into variants, 82 kept apart; 9 proposals sent onto
  existing features. Variants read right ("use aggregation and GROUP BY when needed" under the GROUP BY feature,
  "Fix only defective queries" under "fix the erroneous query").
- The over-clustering finding: three cold-start features are buckets. "understand the question and schema" holds 248
  wordings ("determine necessary joins", "detect hidden constraints including temporal windows", "understand the user's
  intention"), "validate the query before answering" 196, "identify relevant tables and columns" 177 ("retain the top 10
  to 15 most relevant columns", "use the standings tables for cumulative seasonal totals"). The judge raised no misfit
  and no split on them. Two causes, both fixed: the judge prompt called a split rare (a7f7233: a split is the report for
  a bucket, and the reopen keeps the largest part on the node), and the judge saw only a node's sixty best-supported
  members, the generic centre of a bucket, never the one-prompt tail where the other instructions sit (6116ccc: the
  sample spreads across the support range). A first pair of extra rounds with only the prompt fix split 17 small
  features and none of the three buckets; the pair with both fixes runs before the seed.
- 138 standing misfits; the sample reads as real disagreements.

## math v1 (job 12 + extra rounds 16, 18, 19), $8.20 so far
- 8,122 wordings; 588 features + 93 variants in 13 groups; reading coverage 97%; 214 specific after a read.
- Cold start 59 features; round 1 named 409 and folded 70 duplicates at birth; rounds 5 to 13 added 4 to 8 each.
- The join over the run: folds and narrowings in every round (round 2: 18 folded, 4 narrowed of 38 pairs).
- The strong judge split 9 then 6 nodes; after the size rule (a part needs three members) two became variants ("Repeat
  or restate" under "add content beyond the original proof"; "Score or grade the answer" under "evaluate the student's
  response"). The largest nodes, "follow the specified output format" 174 wordings, "reason step by step" 129, "box the
  final answer" 127, were read and left whole: one instruction each. So the split fix is proportionate, not carving.
- Two blemishes for the final refinement: one variant name in Chinese ("精简答案", from before the English rule) and one
  duplicate name, "provide the final answer" twice.

## text2sql after the repair rounds (jobs 13, 15, 17, 20), $7.10 so far
- 391 features + 53 variants; coverage 97%; 125 specific.
- The buckets: "validate the query before answering" (196 wordings) is now a feature of 150 with four variants beneath
  (general review 14, clause and logic 12, schema and types 9, syntax and executability 8); "use tools to explore the
  database" (86) has three (execute and test 15, explore generally 12, inspect sample data 7). "understand the question
  and schema" stays at 209: its split was acted on in an earlier round by reopening, the assigner refiled the members,
  and the flag is standing; one relook would put it through the variant route.
- What the fix did not do: "generate a query that answers the question" (143) and "convert natural language to SQL"
  (137) were read by the strong judge and left whole.

## The seed (job 22), eight rounds, $2.34; the run's last assign was cut off by the account's credits (402)
- 1,200 cards from the four fresh libraries; 122 globals in the 8 aspect groups; 606 cards aligned, 290 domain-specific
  after a read, 304 still open when the credits ran out (they would have had one more assign pass and a final judge).
- Born per round: 39, 23, 20, 12, 8, 8, 6, 6; the join folded 7 duplicates over the run, dropped 2 proposals as topics, and
  ruled on 45 judge pairs (3 folded, the rest kept apart). The seed was stopped once and restarted: its first assign
  rebuilt the whole card set per global per batch (fixed, 591979d).
- Well formed: the middle of the seed reads as one instruction per global across domains ("reason step by step" from
  all four, "format mathematics in LaTeX" from math and science only, "answer in the same language as the user's
  question", "declare variables before use" from math and science). The concentration report tops out at 4%: no global
  holds more than a twenty-fifth of any corpus's features, against job 39's fifth.
- Over-clustering, where it remains: the assign step, not the naming. "present the final answer in the specified format"
  was born from 5 cards and now holds 45 ("put each item on its own line", "format date and time fields according to
  prescribed patterns"); "choose one option from predefined alternatives" 22; "use the provided context or information"
  22; "return only the requested information" 20 ("respond with code only" is filed there). The seed judge flagged 157
  members and the assigner put them all back (all standing), which says the assign prompt's "a narrower feature is not
  an instance" is not being applied by the batch model at low effort on these broad globals. The fix that matches the
  libraries: let the seed judge split, and give the seed level variants so a split becomes structure (allow_variant is
  off at the feature level today); or route the seed's assign of broad globals to the stronger model.
- Under-clustering: 34 indistinct pairs stand as reports; a few read as one instruction ("act as a helpful assistant" ~
  "act as an assistant") and would fold with one more join pass, others are the judge over-reading ("do not provide
  explanations" ~ "do not include units in the final answer").
- Blemishes for the refinement pass: one member card named in Chinese ("不自动扣分省略的规则步骤", from before the English
  rule) and one text2sql feature named "B" (a decomposition artefact) that reached the seed.

## Over- and under-clustering, the libraries
- No bucket: the largest features are the cold start's and carry one instruction each; loop-born features sit at 3 to 16
  wordings with a median of about 6.
- Under-clustering is confined to pairs the loop is not allowed to fold (cold-start pairs) and to pairs reported in the
  final round. Both are visible on the page as reports.
