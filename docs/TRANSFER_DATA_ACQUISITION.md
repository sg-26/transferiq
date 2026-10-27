# Transfer Data Acquisition Contract

Target: next-generation transfer economics model (fee / discount / wage / contract behaviour across LaLiga).

Contract scope: ALL 24 LaLiga clubs committed to 2021/22–2025/26 analytical window (the model's inference seasons are 2022/23–2024/25; 2021/22 and 2025/26 are context-shading seasons). Direction: permanent transfers, loans, loan-with-obligation/option, free agents, undisclosed. RM applies to both **in** and **out** relative to each club.

## 1. One row = one transfer event

Every movement of a player between two clubs that involves a contract change. A loan return is a separate event from the original loan.

## 2. Required fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `transfer_id` | string | yes | stable unique id |
| `tm_player_id` | int | yes | Transfermarkt player id (namespace authority) |
| `player_name` | string | yes | as published by TM |
| `transfer_date` | date | yes | confirmation / registration date |
| `from_club_id` | int | yes | TM club id |
| `from_club` | string | yes | TM club name |
| `to_club_id` | int | yes | TM club id |
| `to_club` | string | yes | TM club name |
| `transfer_type` | categorical | yes | `paid_permanent`, `free`, `loan`, `undisclosed` |
| `fee_eur` | float | conditional | EUR full transfer fee. **NULL when not disclosed — NEVER coerced to 0.** 0 only for true free transfers |
| `fee_type` | categorical | yes | `paid`, `free`, `loan`, `undisclosed` |
| `is_loan` | bool | yes | loan flag (loan-with-obligation still `True`, recorded in `notes`) |
| `loan_duration` | string | optional | e.g. `2025-07-01 -> 2026-06-30` |
| `loan_clause` | categorical | optional | `none`, `purchase_option`, `purchase_obligation` |
| `market_value_at_transfer` | float | yes | TM market value snapshot nearest **before** transfer date |
| `valuation_date` | date | yes | date of that valuation snapshot |
| `source` | string | yes | data source reference |
| `source_url` | string | optional | canonical page |
| `as_of_transfer` | date | yes | capture date of the record |
| `confidence` | float | yes | 0–1 record confidence |
| `notes` | string | optional | free text |

### Fee-type taxonomy (must be preserved exactly)
- `paid` — outright fee > 0 disclosed.
- `free` — no fee, no loan (expired contract / released). fee = 0 **only here**.
- `loan` — temporary; fee optional (loan fee recorded separately if present).
- `undisclosed` — known to be paid but amount not published. fee = NULL.
- `unknown` — whether a fee existed is unconfirmed. fee = NULL. Distinguished from `undisclosed`.

**Rule:** missing/undisclosed/unknown fees are NEVER rewritten to 0. A `0` only ever means documented free transfer.

## 3. Market value at transfer (leakage-safe)

- `market_value_at_transfer` = latest valuation **on or before** the transfer date.
- Never use a future (post-transfer) valuation for a prior event.
- Missing valuation -> NULL with reason code in `notes` (`no_valuation_before_cutoff`).

## 4. Identity contract

Resolution methods, in order of preference; anything else is banned:

1. Exact known TM-awarded ID mapping (e.g. TM player id already present in records).
2. Deterministic mapping (same id resolved by prior manual census).
3. Exact normalized name **+ birth date**.
4. Exact normalized name **+ club + season**.
5. Manually verified.

Banned as final identity keys: fuzzy name matching, surname-only matching, numeric-id equality across namespaces without validation.

Any match below full confidence: `possible_match_only = TRUE` and that record is **never** used in modeling joins until manually verified.

## 5. Current-repo cut (what exists today)

| Dataset | Scope | Rows | Namespace |
|---|---|---|---|
| `data/club_transfers.csv` | Real Madrid only | 105 | TM player ids |
| `data/player_valuations_full.csv` | RM-linked players | 1380 | TM player ids |
| `data/team_full_dossier.csv` | RM checkpoints | 46 | TM player ids |
| `data/laliga_2022_2025/*.parquet` | 24 clubs internal | 1011 players | internal (Wyscout) ids |
| `data/SD-DB(PLAYERS).csv`, `…/La Fabrica…xlsx` | names only | 70, 137 | no usable ids |

The curated TM-id -> internal-team-id club crosswalk used in Phase 0/1 is at `C:\football\src\data\validate_transfer_data.py` (`_TM_CLUB_IDS`) and the registry in `data/processed/players_registry.parquet`.

## 6. External acquisition plan (do NOT auto-download large sets)

Acquire in this order (each is a bounded, reviewable batch):

1. **TM transfer history per club** — for the 23 non-RM LaLiga clubs 2021/22–2025/26: transfers page per club-season (in/out). ~23 clubs × 5 seasons × 2 directions ≈ 230 page loads. Provider: Transfermarkt. Fields from §2.
2. **TM market-value history per target player** — for every new non-RM player: valuation timeline to build `market_value_at_transfer`. Provider: TM.
3. **ID crosswalk** — TM player id ↔ internal (Wyscout/current) id for the ~23 clubs' squads. Sources: TM player pages, club squad pages, manual census. This is the identity bottleneck (see coverage report, §7).
4. **Loan clause confirmation** — option / obligation / straight loan. TM loan pages; secondary: official club statements (captured as URLs).

Rejected as auto-fetch: bulk scraping of TM all-clubs, gambling/agent fee-aggregator mirrors, and any dataset mixing namespaces without validation.

## 7. Achievement threshold

Coverage target: ≥80% of delivered permanent-deal rows carry a disclosed or genuinely-free fee with a leakage-safe `market_value_at_transfer`. Loans and undisclosed rows are counted separately and never blended into `fee_eur`.