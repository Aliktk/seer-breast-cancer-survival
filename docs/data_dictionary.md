# Data Dictionary

Source file: `data/raw/SEER.csv` (4,024 rows × 16 columns, one of them empty).
Cleaned file: `data/interim/seer_clean.csv` (produced by notebook 02).

| Raw column | Clean name | Type | Values in data | Role | Notes |
|---|---|---|---|---|---|
| `Age` | `age` | int | 30–69 | input | age at diagnosis, years |
| `Race ` | `race` | category | White (3,413), Black (291), Other (320) | input | "Other" = American Indian/AK Native, Asian/Pacific Islander. Trailing space in raw header |
| `Marital Status` | `marital_status` | category | Married (2,643), Single (615), Divorced (486), Widowed (235), Separated (45) | input | |
| `Unnamed: 3` | – | – | 100 % empty | dropped | |
| `T Stage ` | `t_stage` | ordinal | T1 (1,603), T2 (1,786), T3 (533), T4 (102) | input | AJCC 6th ed. Trailing space in raw header |
| `N Stage` | `n_stage` | ordinal | N1 (2,732), N2 (820), N3 (472) | input | all patients are node-positive |
| `6th Stage` | `stage_6th` | ordinal | IIA, IIB, IIIA, IIIB, IIIC | **excluded** | 100 % determined by T and N |
| `Grade` | `grade` | ordinal 1–4 | 1 (543), 2 (2,351), 3 (1,111), 4 (19) | input | text labels mapped to integers |
| `A Stage` | `a_stage` | binary | Regional (3,932), Distant (92) | input | SEER summary stage |
| `Tumor Size` | `tumor_size` | int (mm) | 1–140 | input | right-skewed (skew ≈ 1.7) |
| `Estrogen Status` | `estrogen_status` | binary | Positive (3,755), Negative (269) | input | |
| `Progesterone Status` | `progesterone_status` | binary | Positive (3,326), Negative (698) | input | |
| `Regional Node Examined` | `nodes_examined` | int | 1–61 | input | |
| `Reginol Node Positive` | `nodes_positive` | int | 1–46 | input | typo in raw header |
| `Survival Months` | `survival_months` | int | 1–107 | **outcome** | time to death or last follow-up. Never an input |
| `Status` | `status` | binary | Alive (3,408), Dead (616) | **outcome** | all-cause vital status |
| – | `event` | 0/1 | 616 events | **outcome** | 1 = Dead |

## Engineered features (`src/seer_survival/features.py`)

| Feature | Definition |
|---|---|
| `node_ratio` | `nodes_positive / nodes_examined` |
| `log_tumor_size` | `log(1 + tumor_size)` |
| `log_nodes_positive` | `log(1 + nodes_positive)` |
| `er_positive`, `pr_positive` | 1 if receptor positive |
| `distant_a_stage` | 1 if `a_stage == "Distant"` |
| `hr_status` | joint ER/PR: `ER+/PR+`, `ER+/PR-`, `ER-/PR+`, `ER-/PR-` |

## Consistency flags (`data/interim/seer_clean.csv`)

| Flag | Rule (AJCC 6th ed.) | Rows |
|---|---|---|
| `flag_t1_size` | T1 with size > 20 mm | 0 |
| `flag_t2_size` | T2 with size outside 21–50 mm | 0 |
| `flag_t3_size` | T3 with size ≤ 50 mm | 7 |
| `flag_n_count` | N1 with > 3, N2 outside 4–9, or N3 with < 10 positive nodes | 101 |
| any flag | | 107 |
