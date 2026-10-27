# Transfer Data Coverage Report

Produced from the repository data layer (`data/processed/transfers.parquet`, `data/processed/players_registry.parquet`, `data/processed/identity_review.csv`) validated by `src/data/validate_transfer_data.py` (15/15 PASS).

## 1. Overall coverage

| Metric | Value |
|---|---|
| Clubs covered | **1 of 24** (Real Madrid only) |
| Analytical seasons usable | 2022/23–2024/25 (of 5 in scope) |
| Transfer events (usable) | **105** |
| Directonal split | 53 in / 52 out (vs RM) |
| Paid permanent | 31 |
| Free | 16 |
| Loan | 37 |
| Undisclosed/unknown | 21 |
| Transfer fee present (`fee_eur` non-null, non-loan) | 47 |
| Leakage-safe `market_value_at_transfer` | 105 / 105 (100%: 60 via pvf asof, 45 via source MV) |
| Player-id resolved to internal | 48 / 105 events (via registry bridges) |

## 2. Coverage by season

| Season | Rows | Trend |
|---|---|---|
| 2021/2022 | 11 | (context only) |
| 2022/2023 | 22 | first analytical |
| 2023/2024 | 23 | |
| 2024/2025 | 22 | last analytical |
| 2025/2026 | 14 | future/context |
| 2026/2027 | 13 | future/context |

Analytical window completeness for RM: playset present but **the other 23 clubs have zero rows**.

## 3. Coverage by direction & type

| | in | out |
|---|---|---|
| paid_permanent | 13 | 18 |
| free | 4 | 12 |
| loan | 20 | 17 |
| undisclosed_or_unknown | 16 | 5 |

Out-transfers dominate paid/free (sell-side visible); undisclosed rows concentrated in Castilla-to-first-team promotions (RM-internal, fee never published).

## 4. Undisclosed / missing-fee honesty

21 rows have `fee_eur = NULL` and are **not** loans: all are `undisclosed_or_unknown` (e.g., Castilla promotions, Hazard/Kroos/Carvajal departures). Zero fees are exercised to infer a paid transfer. The validator enforces this invariant (`missing_fee_not_zero` PASS).

## 5. Identity resolution coverage

| Method | Rows in registry | Possible-match-only? |
|---|---|---|
| internal_spine (no TM id) | 967 | no |
| norm_name_club_season (method-4) | 33 | no (confident) |
| norm_name_only_provisional | 11 | yes |
| unresolved (no internal candidate) | 22 | yes |

- Confident bridges: 33 (both ids present, method-4, used in transfers joins).
- Provisional/unresolved: 33 TM side records never used as join keys (`possible_match_only=TRUE`).
- 22 unresolved TM records include: Kylian Mbappé, Marcelo, Bernardo Silva, Marc Cucurella, Dean Huijsen, Trent Alexander-Arnold, Denzel Dumfries, Ibrahima Konaté, Sergio Arribas, Franco Mastantuono — either absent from the internal spine or internal full-name differs from TM full-name (e.g. "David Olatukunbo Alaba" vs "David Alaba").

## 6. Source coverage by club / season / direction / type

Only RM events exist; therefore club/season/direction/type matrices beyond RM are structurally empty. Provisionally, RM coverage looks complete for the analytical window (105 rows spanning 21/22–24/25 with 59 distinct TM player ids) but has **no independent second source** to confirm completeness (no cross-check vs an external transfer list).

## 7. Critical question: is ≥80% permanent-deal coverage achievable with current repo data?

**No, for cross-club scope.** Current data gives 31 paid-permanent + 16 free events, all RM-side, so full 24-club permanent-deal coverage is 0/any target. For the RM-only slice, disclosed/non-loan fee coverage is 47/105 rows (45%), well under 80% primarily because **21 undisclosed** and **37 loans** are legitimately excluded; if the numerator is restricted to *paid-permanent* the disclosed fee rate is 31/31 (100%), but the denominator is one club and ~70 deals.

Realistic path to ≥80%: acquire TM per-club transfer tables (see TRANSFER_DATA_ACQUISITION.md §6). With the 23 non-RM clubs added, permanent-deal counts rise into the high hundreds; disclosed-fee share then depends on TM publication, historically >80% for guarantees and >50% for total including add-ons. Expectation after acquisition is documented below.

## 8. Expected coverage after full acquisition (projection)

| Metric | Projected |
|---|---|
| Clubs | 24/24 |
| Permanent deals (21/22–25/26) | ~400–500 all-club |
| Disclosed-fee share (paid) | ~80–90% |
| Leakage-safe MV at transfer | ~90–100% (per-player valuation timelines) |
| Identity-resolved (TM→internal) | needs the manual crosswalk census (§6.3); name+club+season alone estimated 60–75% |

## 9. What to acquire first

1. TM per-club transfers (23 clubs off-RM), 21/22–25/26, in+out — this kills the single-club skew.
2. Valuation timeline per covered player for leakage-safe `market_value_at_transfer`.
3. The TM→internal id crosswalk census for the 23 clubs (identity bottleneck).
4. Loan-clause confirmation pass.