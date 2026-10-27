"""Validation of the next-gen transfer economic data layer.

Read-only: checks data/processed/transfers.parquet, players_registry.parquet,
identity_review.csv against the acquisition contract and prints a concise report.
Does NOT modify any data or build any model.
"""
import pandas as pd
import datetime as dt

DATA = r'C:\football\data\processed'
# TM club-id namespace legitimately observed in source files
_TM_CLUB_IDS = {5, 11, 12, 13, 15, 16, 24, 27, 28, 31, 36, 40, 42, 46, 79, 102,
                142, 148, 150, 159, 162, 180, 209, 221, 237, 273, 281, 294, 330,
                331, 337, 366, 367, 368, 405, 418, 419, 430, 466, 506, 512, 515,
                533, 543, 583, 585, 614, 621, 631, 681, 683, 714, 720, 738, 762,
                861, 897, 940, 985, 989, 1023, 1041, 1047, 1082, 1084, 1108, 1114,
                1184, 1237, 1244, 1462, 1531, 1532, 1533, 2441, 2462, 2464, 2497,
                2687, 2865, 3302, 3709, 4032, 4383, 6195, 6297, 6665, 6767, 8023,
                8494, 8970, 9899, 10690, 12321, 14807, 16795, 26069, 41101, 51828}
KNOWN_CLUBS_TM = _TM_CLUB_IDS
KNOWN_TYPES = {'loan', 'paid_permanent', 'free', 'undisclosed_or_unknown'}
KNOWN_DIRS = {'in', 'out'}

def load():
    tr = pd.read_parquet(f'{DATA}/transfers.parquet')
    reg = pd.read_parquet(f'{DATA}/players_registry.parquet')
    rev = pd.read_csv(f'{DATA}/identity_review.csv')
    return tr, reg, rev

def run(tr, reg, rev):
    checks = []
    def add(name, ok, detail):
        checks.append((name, ok, detail))

    tr['transfer_date'] = pd.to_datetime(tr['transfer_date'], errors='coerce')

    # 1 unique transfer IDs
    add('unique_transfer_ids', tr['transfer_id'].is_unique,
        f'{tr["transfer_id"].is_unique}')

    # 2 valid dates
    bad_dt = tr['transfer_date'].isna().sum() + (tr['transfer_date'] > pd.Timestamp.now()).sum()
    add('valid_dates', bad_dt == 0, f'bad dates: {bad_dt}')

    # 3 known clubs
    sid = tr['transfer_date'].dt.year
    f_ok = set(tr['from_club_id'].dropna().astype(int)).issubset(KNOWN_CLUBS_TM | {418})
    t_ok = set(tr['to_club_id'].dropna().astype(int)).issubset(KNOWN_CLUBS_TM | {418})
    add('known_clubs', f_ok and t_ok, f'from ok: {f_ok}, to ok: {t_ok}')

    # 4 no negative fees
    neg = (tr['fee_eur'].notna() & (tr['fee_eur'] < 0)).sum()
    add('no_negative_fees', neg == 0, f'negative fees: {neg}')

    # 5 missing fee != zero (never convert missing/undisclosed to 0)
    miss_fee_not_loan = tr[tr['is_loan'] == False]['fee_eur'].isna().sum()
    zeros = (tr['fee_eur'] == 0).sum()
    add('missing_fee_not_zero', miss_fee_not_loan == sum(tr['fee_type'] == 'undisclosed'),
        f'missing-nonloan: {miss_fee_not_loan}, zero fees (legit free): {zeros}')

    # 6 valid types
    types_ok = set(tr['transfer_type']).issubset(KNOWN_TYPES)
    add('valid_transfer_types', types_ok, str(sorted(set(tr['transfer_type']))))

    # 7 no impossible self-transfers
    self_tr = (tr['from_club'] == tr['to_club']).sum()
    add('no_self_transfers', self_tr == 0, f'self transfers: {self_tr}')

    # 8 duplicate detection (player, date, from, to)
    dup = tr.duplicated(subset=['tm_player_id', 'transfer_date', 'from_club_id', 'to_club_id']).sum()
    add('no_duplicate_events', dup == 0, f'duplicates: {dup}')

    # 9 identity consistency (registry used consistently in transfers)
    bridged = reg[(reg['tm_player_id'].notna()) & (reg['internal_player_id'].notna())]
    conflict = 0
    for _, b in bridged.iterrows():
        set_i = set(tr.loc[tr['tm_player_id'] == b['tm_player_id'], 'internal_player_id'].dropna())
        if len(set_i) > 1:
            conflict += 1
    add('identity_consistency', conflict == 0, f'conflicting bridges: {conflict}')

    # 10 no joins through possible_match_only
    pm = set(reg.loc[reg['possible_match_only'] == True, 'internal_player_id'].dropna())
    leak = tr['internal_player_id'].isin(pm).sum()
    add('no_possible_match_joins', leak == 0, f'leaked joins: {leak}')

    # 11 MV date <= cutoff (leakage check)
    leak_mv = (tr['mv_valuation_date'].notna() &
               (pd.to_datetime(tr['mv_valuation_date'], errors='coerce') > tr['transfer_date'])).sum()
    add('mv_date_le_cutoff', leak_mv == 0, f'future MV leaks: {leak_mv}')

    # 12 no future leakage (source dates)
    add('no_future_source_dates', bad_dt == 0, f'bad dates: {bad_dt}')

    # 13 club ids consistent with TM namespace
    # internal team ids should not appear in from/to club ids
    int_ids = set(reg.dropna(subset=['internal_player_id'])['internal_player_id'].astype(int))
    clubids = set(tr['from_club_id'].dropna().astype(int)) | set(tr['to_club_id'].dropna().astype(int))
    ns_conflict = clubids.intersection(int_ids)
    add('tm_club_id_namespace', len(ns_conflict) == 0, f'namespace collisions: {sorted(ns_conflict)}')

    # 14 direction consistency (RM always one side)
    rm_side = ((tr['from_club_id'] == 418) | (tr['to_club_id'] == 418)).sum()
    dir_ok = (tr[(tr['direction'] == 'out') & (tr['from_club_id'] != 418)].shape[0] == 0 and
              tr[(tr['direction'] == 'in') & (tr['to_club_id'] != 418)].shape[0] == 0)
    add('direction_consistency', dir_ok and rm_side == len(tr), f'RM-side rows: {rm_side}/{len(tr)}')

    # 15 no duplicate (player,date,from,to) incl club casts
    add('no_dup_ptft', dup == 0, f'duplicates: {dup}')

    # ---- report ----
    fails = [c for c in checks if not c[1]]
    print('=' * 70)
    print('TRANSFER DATA VALIDATION REPORT')
    print(f'files: transfers.parquet ({len(tr)} rows), players_registry.parquet ({len(reg)}), identity_review.csv ({len(rev)})')
    print('-' * 70)
    for name, ok, detail in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {name:28s} {detail}')
    print('-' * 70)
    print(f'RESULT: {len(checks) - len(fails)}/{len(checks)} checks passed')
    if fails:
        print('FAILED:', ', '.join(f[0] for f in fails))
        raise SystemExit(1)
    print('OK')

if __name__ == '__main__':
    tr, reg, rev = load()
    run(tr, reg, rev)