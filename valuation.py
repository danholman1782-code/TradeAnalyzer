
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# ---------- Scoring ----------
@dataclass
class ScoringSettings:
    # Passing
    pass_yd_pt: float = 0.04
    pass_td: float = 4
    pass_td50_bonus: float = 4
    int_thrown: float = -2
    pass_2pt: float = 2
    pass_400g_bonus: float = 5
    # Rushing
    rush_yd_pt: float = 0.1
    rush_td: float = 6
    rush_td50_bonus: float = 4
    rush_2pt: float = 2
    rush_100g_bonus: float = 5
    rush_200g_bonus: float = 10
    # Receiving
    rec_yd_pt: float = 0.1
    reception: float = 1.0
    rec_td: float = 6
    rec_td50_bonus: float = 4
    rec_2pt: float = 2
    rec_100g_bonus: float = 5
    rec_200g_bonus: float = 10
    # Kicking
    pat_made: float = 1
    fg_missed: float = -1
    fg_0_39: float = 3
    fg_40_49: float = 4
    fg_50_59: float = 5
    fg_60_plus: float = 5
    # D/ST categories
    dst_kickret_td: float = 6
    dst_puntret_td: float = 6
    dst_intret_td: float = 6
    dst_fumret_td: float = 6
    dst_blockret_td: float = 6
    dst_2pt_ret: float = 2
    dst_1pt_safety: float = 1
    dst_sack: float = 1
    dst_block_kick: float = 2
    dst_int: float = 2
    dst_fr: float = 2
    dst_safety: float = 2
    # Points Allowed buckets (counts of games)
    pa0: float = 10
    pa1_6: float = 8
    pa7_13: float = 5
    pa14_17: float = 3
    pa28_34: float = -1
    pa35_45: float = -3
    pa46_plus: float = -5
    # Yards Allowed buckets (counts of games)
    ya_lt100: float = 5
    ya100_199: float = 3
    ya200_299: float = 2
    ya350_399: float = -1
    ya400_449: float = -3
    ya450_499: float = -5
    ya500_549: float = -6
    ya550_plus: float = -7

# ---------- League / Dynasty / Keeper ----------
@dataclass
class LeagueSettings:
    teams: int
    roster_starters: Dict[str, int]
    ppr: float = 1.0
    te_premium: float = 0.0
    qb_passing_td: int = 4
    superflex: bool = False
    scoring: ScoringSettings = field(default_factory=ScoringSettings)  # FIX: default_factory

@dataclass
class DynastySettings:
    enabled: bool = False
    window_years: int = 3
    discount_rate: float = 0.12
    win_now_bias: float = 0.0
    prime_age: Dict[str, int] = None
    decline_start: Dict[str, int] = None
    def __post_init__(self):
        self.prime_age = self.prime_age or {"QB": 29, "RB": 24, "WR": 26, "TE": 27, "DEF": 0, "K": 0}
        self.decline_start = self.decline_start or {"QB": 34, "RB": 27, "WR": 29, "TE": 30, "DEF": 0, "K": 0}

@dataclass
class KeeperSettings:
    enabled: bool = False
    round_step: int = 2
    min_round: int = 1
    draft_rounds: int = 16
    undrafted_round: Optional[int] = None
    min_keep_round: int = 3           # blocks R1/R2
    disallowed_rounds: Optional[List[int]] = None
    include_in_trade_score: bool = False
    keeper_weight: float = 1.0
    keeper_slots_team_a: Optional[int] = 4  # Keep-4 cap
    keeper_slots_team_b: Optional[int] = 4

@dataclass
class Trade:
    team_a_gives: List[str]
    team_b_gives: List[str]
    team_a_picks_gives: Optional[List[str]] = None
    team_b_picks_gives: Optional[List[str]] = None

# ---------- Scoring pipeline ----------
def apply_scoring(df: pd.DataFrame, ls: LeagueSettings) -> pd.DataFrame:
    s = ls.scoring
    out = df.copy()
    out["pos"] = out["pos"].astype(str).str.upper()
    def g(col):
        return out[col] if col in out.columns else 0.0
    te_ppr = s.reception + ls.te_premium
    pass_pts = g("proj_pass_yd")*s.pass_yd_pt + g("proj_pass_td")*s.pass_td + g("proj_pass_td50")*s.pass_td50_bonus + g("proj_int")*s.int_thrown + g("proj_pass_2pt")*s.pass_2pt + g("proj_400p_games")*s.pass_400g_bonus
    rush_pts = g("proj_rush_yd")*s.rush_yd_pt + g("proj_rush_td")*s.rush_td + g("proj_rush_td50")*s.rush_td50_bonus + g("proj_rush_2pt")*s.rush_2pt + g("proj_100r_games")*s.rush_100g_bonus + g("proj_200r_games")*s.rush_200g_bonus
    rec_pts = g("proj_rec")*te_ppr + g("proj_rec_yd")*s.rec_yd_pt + g("proj_rec_td")*s.rec_td + g("proj_rec_td50")*s.rec_td50_bonus + g("proj_rec_2pt")*s.rec_2pt + g("proj_100re_games")*s.rec_100g_bonus + g("proj_200re_games")*s.rec_200g_bonus
    k_pts = g("proj_pat_made")*s.pat_made + g("proj_fg_missed")*s.fg_missed + g("proj_fg_0_39_made")*s.fg_0_39 + g("proj_fg_40_49_made")*s.fg_40_49 + g("proj_fg_50_59_made")*s.fg_50_59 + g("proj_fg_60_plus_made")*s.fg_60_plus
    dst_cat = g("proj_dst_sacks")*s.dst_sack + g("proj_dst_blocked_kicks")*s.dst_block_kick + g("proj_dst_int")*s.dst_int + g("proj_dst_fr")*s.dst_fr + g("proj_dst_safeties")*s.dst_safety + g("proj_dst_kickret_td")*s.dst_kickret_td + g("proj_dst_puntret_td")*s.dst_puntret_td + g("proj_dst_intret_td")*s.dst_intret_td + g("proj_dst_fumret_td")*s.dst_fumret_td + g("proj_dst_blockret_td")*s.dst_blockret_td + g("proj_dst_2pt_ret")*s.dst_2pt_ret + g("proj_dst_1pt_safety")*s.dst_1pt_safety
    pa = g("proj_pa0_games")*s.pa0 + g("proj_pa1_6_games")*s.pa1_6 + g("proj_pa7_13_games")*s.pa7_13 + g("proj_pa14_17_games")*s.pa14_17 + g("proj_pa28_34_games")*s.pa28_34 + g("proj_pa35_45_games")*s.pa35_45 + g("proj_pa46_plus_games")*s.pa46_plus
    ya = g("proj_ya_lt100_games")*s.ya_lt100 + g("proj_ya100_199_games")*s.ya100_199 + g("proj_ya200_299_games")*s.ya200_299 + g("proj_ya350_399_games")*s.ya350_399 + g("proj_ya400_449_games")*s.ya400_449 + g("proj_ya450_499_games")*s.ya450_499 + g("proj_ya500_549_games")*s.ya500_549 + g("proj_ya550_plus_games")*s.ya550_plus
    out["proj_pts_raw"] = pass_pts + rush_pts + rec_pts
    out.loc[out["pos"]=="K","proj_pts_raw"] = k_pts[out["pos"]=="K"]
    out.loc[out["pos"].isin(["DEF","DST"]),"proj_pts_raw"] = (dst_cat + pa + ya)[out["pos"].isin(["DEF","DST"])]
    return out

# ---------- Replacement & VORP ----------
def replacement_threshold(ls: LeagueSettings) -> Dict[str, int]:
    base = {"QB": ls.roster_starters.get("QB",0)*ls.teams,
            "RB": ls.roster_starters.get("RB",0)*ls.teams,
            "WR": ls.roster_starters.get("WR",0)*ls.teams,
            "TE": ls.roster_starters.get("TE",0)*ls.teams,
            "DEF": ls.roster_starters.get("DEF",0)*ls.teams,
            "K": ls.roster_starters.get("K",0)*ls.teams}
    flex = ls.roster_starters.get("FLEX",0)*ls.teams
    sflex = ls.roster_starters.get("SUPERFLEX",0)*ls.teams
    base["WR"] += int(round(flex*0.6)); base["RB"] += int(round(flex*0.3)); base["TE"] += int(round(flex*0.1))
    base["QB"] += int(round(sflex*0.5)); base["WR"] += int(round(sflex*0.25)); base["RB"] += int(round(sflex*0.25))
    return base


def vorp_by_position(df: pd.DataFrame, ls: LeagueSettings) -> pd.DataFrame:
    df = df.copy()
    repl = replacement_threshold(ls)
    frames = []
    repl_pts: Dict[str, float] = {}
    for pos, grp in df.groupby("pos"):
        grp = grp.sort_values("proj_pts_raw", ascending=False).reset_index(drop=True)
        idx = max(repl.get(pos,0)-1, 0)
        if len(grp)==0: continue
        repl_pt = grp.loc[idx, "proj_pts_raw"] if idx < len(grp) else grp.iloc[-1]["proj_pts_raw"]
        repl_pts[pos] = float(repl_pt)
        frames.append(grp)
    out = pd.concat(frames, axis=0) if frames else df
    out["replacement_pts"] = out["pos"].map(lambda p: repl_pts.get(p,0.0))
    out["vorp"] = out["proj_pts_raw"] - out["replacement_pts"]
    med = out.groupby("pos")["vorp"].median().to_dict()
    def boost(r):
        m = med.get(r["pos"],0.0)
        return r["vorp"] * (1.10 if m<=0 else (1.0 + min(0.10, 0.5/(m+0.01))))
    out["vorp_adj"] = out.apply(boost, axis=1)
    risk = out.get("risk", pd.Series(1.0, index=out.index)).clip(0.7, 1.2)
    out["value_redraft"] = out["vorp_adj"] * risk
    return out.sort_values(["pos","value_redraft"], ascending=[True,False])

# ---------- Dynasty ----------
def _age_multiplier(pos: str, age: float, ds: DynastySettings) -> float:
    if age is None or (isinstance(age,float) and np.isnan(age)) or pos in ("DEF","DST","K"):
        base = 1.0
    else:
        prime = ds.prime_age.get(pos,26); decline = ds.decline_start.get(pos, prime+3)
        if age <= prime: base = 0.9 + 0.02*min(5.0, max(0.0, prime-age))
        else: base = 1.0 - 0.06*max(0.0, age-decline)
        base = float(np.clip(base, 0.6, 1.2))
    bias = ds.win_now_bias
    if bias != 0 and age is not None and not (isinstance(age,float) and np.isnan(age)) and pos not in ("DEF","DST","K"):
        prime = ds.prime_age.get(pos,26)
        norm = max(-4.0, min(4.0, age-prime))/4.0
        base *= (1.0 + 0.10*bias*norm)
    return float(np.clip(base, 0.5, 1.3))


def _positional_decay(pos: str) -> float:
    return {"QB":0.03, "WR":0.07, "TE":0.08, "RB":0.15, "DEF":0.10, "K":0.05}.get(pos, 0.08)


def dynasty_values(df_values: pd.DataFrame, ls: LeagueSettings, ds: DynastySettings) -> pd.DataFrame:
    if not ds.enabled:
        out = df_values.copy(); out["value_dynasty"] = out["value_redraft"]; return out
    df = df_values.copy()
    ages = df.get("age", pd.Series([np.nan]*len(df), index=df.index))
    base = df["vorp_adj"].clip(lower=0)
    risk = df.get("risk", pd.Series(1.0, index=df.index)).clip(0.7, 1.2)
    total = np.zeros(len(df), dtype=float)
    for year in range(1, ds.window_years+1):
        pv = 1.0 / ((1.0 + ds.discount_rate) ** (year-1))
        tilt = 1.0 + (ds.win_now_bias * 0.10 * (1.0 - (year-1)/max(1, ds.window_years-1)))
        decay = df["pos"].map(lambda p: (1.0 - _positional_decay(p)) ** (year-1))
        age_mult = []
        for i,row in df.iterrows():
            age = ages.loc[i]
            adj = age + (year-1) if pd.notna(age) else np.nan
            age_mult.append(_age_multiplier(row["pos"], adj, ds))
        total += base.values * risk.values * pv * tilt * decay.values * np.array(age_mult)
    df["value_dynasty"] = total
    return df

# ---------- Keepers ----------

def _ensure_undrafted_round(ks: KeeperSettings) -> int:
    return ks.undrafted_round if ks.undrafted_round and ks.undrafted_round>0 else ks.draft_rounds

def keeper_next_round(draft_round: Optional[float], years_kept: Optional[int], ks: KeeperSettings) -> int:
    base_round = int(draft_round) if (draft_round and draft_round>0) else _ensure_undrafted_round(ks)
    kept = int(years_kept) if years_kept is not None else 0
    return int(max(ks.min_round, base_round - ks.round_step*max(0, kept+1)))


def _expected_value_by_round(df_sorted: pd.DataFrame, ls: LeagueSettings, value_col: str, ks: KeeperSettings) -> Dict[int,float]:
    values = {}
    for rnd in range(1, ks.draft_rounds+1):
        start, end = (rnd-1)*ls.teams, rnd*ls.teams
        sl = df_sorted.iloc[start:end]
        values[rnd] = float(sl[value_col].mean()) if len(sl) else 0.0
    return values


def apply_keeper_model(df_values: pd.DataFrame, ls: LeagueSettings, ds: DynastySettings, ks: KeeperSettings) -> pd.DataFrame:
    df = df_values.copy()
    if not ks.enabled:
        df["keeper_next_round"] = np.nan; df["keeper_expected_value_next_round"] = np.nan
        df["keeper_surplus"] = 0.0; df["keeper_eligible"] = False; return df
    value_col = "value_dynasty" if ds.enabled else "value_redraft"
    df = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    df["rank_overall"] = np.arange(1, len(df)+1)
    if "draft_round" not in df.columns: df["draft_round"] = df["rank_overall"].apply(lambda r: int(np.ceil(r/ls.teams)))
    if "years_kept" not in df.columns: df["years_kept"] = 0
    blocked = set(ks.disallowed_rounds or []) | set(range(1, ks.min_keep_round))
    def eligible(r):
        if pd.isna(r): return False
        rr = int(r); return (rr not in blocked) and (rr >= ks.min_keep_round)
    df["keeper_eligible"] = df["draft_round"].apply(eligible)
    df.loc[df["keeper_eligible"], "keeper_next_round"] = df.loc[df["keeper_eligible"]].apply(lambda r: keeper_next_round(r.get("draft_round",np.nan), r.get("years_kept",0), ks), axis=1)
    df.loc[~df["keeper_eligible"], "keeper_next_round"] = pd.NA
    expected_map = _expected_value_by_round(df_sorted=df, ls=ls, value_col=value_col, ks=ks)
    df["keeper_expected_value_next_round"] = df["keeper_next_round"].map(lambda rnd: expected_map.get(int(rnd),0.0) if pd.notna(rnd) else 0.0)
    df["keeper_surplus"] = 0.0
    elig = df["keeper_eligible"].fillna(False)
    df.loc[elig, "keeper_surplus"] = df.loc[elig, value_col] - df.loc[elig, "keeper_expected_value_next_round"]
    df["keeper_surplus"] = df["keeper_surplus"].clip(lower=0.0)
    return df

# ---------- Picks & Trade ----------

def _parse_pick(pick: str) -> Tuple[int,int,int]:
    s = pick.strip().upper().replace('-', ' ').replace('.', ' ')
    parts = [p for p in s.split() if p]
    if len(parts) < 2: raise ValueError(f"Unrecognized pick format: {pick}")
    year = int(parts[0])
    if len(parts) == 2:
        tok = parts[1]
        if tok.isdigit(): rnd, sel = int(tok), 1
        else: rnd, sel = int(tok[0]), int(tok[1:])
    else:
        rnd, sel = int(parts[1]), int(parts[2])
    return year, rnd, sel


def _base_pick_curve(superflex: bool) -> Dict[Tuple[int,int], float]:
    r1 = [60,56,53,50,48,46,44,42,40,38,36,34]
    r2 = [28,27,26,25,24,23,22,21,20,19,18,17]
    r3 = [12,11,10,9,9,8,8,7,7,6,6,5]
    r4 = [4,4,3,3,3,2.5,2.5,2,2,1.5,1.5,1.0]
    curve = {(1,i+1):v for i,v in enumerate(r1)}; curve.update({(2,i+1):v for i,v in enumerate(r2)})
    curve.update({(3,i+1):v for i,v in enumerate(r3)}); curve.update({(4,i+1):v for i,v in enumerate(r4)})
    if superflex:
        for sel in range(1,7): curve[(1,sel)] *= 1.25
        for sel in range(7,13): curve[(1,sel)] *= 1.10
        for sel in range(1,5): curve[(2,sel)] *= 1.05
    return curve


def value_future_pick(pick: str, ls: LeagueSettings, ds: DynastySettings, current_year: int=None) -> float:
    year, rnd, sel = _parse_pick(pick)
    base = _base_pick_curve(ls.superflex).get((rnd, sel), 0.5)
    if current_year is None: current_year = pd.Timestamp.today().year
    years_out = max(0, year - current_year)
    disc = ds.discount_rate if ds.enabled else 0.12
    pv = 1.0/((1.0+disc)**years_out)
    return float(base*pv)


def trade_value(
    df_values: pd.DataFrame,
    trade: Trade,
    team_a_roster_counts: Optional[Dict[str,int]] = None,
    team_b_roster_counts: Optional[Dict[str,int]] = None,
    ls: Optional[LeagueSettings] = None,
    ds: Optional[DynastySettings] = None,
    ks: Optional[KeeperSettings] = None,
    mode: str = "redraft",
) -> Dict:
    # SAFE defaults (avoid mutable defaults)
    ds = ds or DynastySettings(enabled=False)
    ks = ks or KeeperSettings(enabled=False)
    value_col = "value_dynasty" if (ds.enabled and mode=="dynasty") else "value_redraft"

    df = df_values.set_index("name", drop=False)
    missing = [p for p in (trade.team_a_gives + trade.team_b_gives) if p not in df.index]
    if missing:
        return {"error": f"Players not found in dataset: {missing}"}

    caps = replacement_threshold(ls) if ls else {"QB":1,"RB":2,"WR":2,"TE":1,"DEF":1,"K":0}
    team_a_roster_counts = team_a_roster_counts or {k:0 for k in caps}
    team_b_roster_counts = team_b_roster_counts or {k:0 for k in caps}

    def eff(row, counts):
        pos = row["pos"]; need = caps.get(pos,0); have = counts.get(pos,0)
        return float(row[value_col] * (0.6 if have>=need else 1.0))

    a_in = df.loc[trade.team_b_gives]; a_out = df.loc[trade.team_a_gives]
    b_in = df.loc[trade.team_a_gives]; b_out = df.loc[trade.team_b_gives]

    a_in_val = a_in.apply(lambda r: eff(r, team_a_roster_counts), axis=1).sum(); a_out_val = a_out[value_col].sum()
    b_in_val = b_in.apply(lambda r: eff(r, team_b_roster_counts), axis=1).sum(); b_out_val = b_out[value_col].sum()

    picks_detail = {"team_a_in_picks":[],"team_a_out_picks":[],"team_b_in_picks":[],"team_b_out_picks":[]}
    if mode=="dynasty":
        now = pd.Timestamp.today().year
        a_out_p = trade.team_a_picks_gives or []; b_out_p = trade.team_b_picks_gives or []
        a_out_val_p = sum(value_future_pick(p, ls, ds, now) for p in a_out_p)
        b_out_val_p = sum(value_future_pick(p, ls, ds, now) for p in b_out_p)
        a_in_val += b_out_val_p; a_out_val += a_out_val_p
        b_in_val += a_out_val_p; b_out_val += b_out_val_p
        picks_detail["team_a_in_picks"] = [{"pick": p, "value": value_future_pick(p, ls, ds, now)} for p in b_out_p]
        picks_detail["team_a_out_picks"] = [{"pick": p, "value": value_future_pick(p, ls, ds, now)} for p in a_out_p]
        picks_detail["team_b_in_picks"] = [{"pick": p, "value": value_future_pick(p, ls, ds, now)} for p in a_out_p]
        picks_detail["team_b_out_picks"] = [{"pick": p, "value": value_future_pick(p, ls, ds, now)} for p in b_out_p]

    # Keeper deltas (Keep-4 cap)
    kda = 0.0; kdb = 0.0
    if ks.enabled and ks.include_in_trade_score:
        if "keeper_surplus" not in df.columns: df["keeper_surplus"] = 0.0
        if "keeper_eligible" not in df.columns: df["keeper_eligible"] = False
        def sum_surplus(rows, slots):
            if rows.empty: return 0.0
            sub = rows[rows.get("keeper_eligible", False) == True]
            if sub.empty: return 0.0
            s = sub["keeper_surplus"].clip(lower=0.0).sort_values(ascending=False)
            if slots is None: return float(s.sum())
            if slots <= 0: return 0.0
            return float(s.head(slots).sum())
        kda += sum_surplus(a_in, ks.keeper_slots_team_a); kdb += sum_surplus(b_in, ks.keeper_slots_team_b)
        kda -= sum_surplus(a_out, ks.keeper_slots_team_a); kdb -= sum_surplus(b_out, ks.keeper_slots_team_b)

    a_core = float(a_in_val - a_out_val); b_core = float(b_in_val - b_out_val)
    a_tot = a_core + (ks.keeper_weight*kda if (ks.enabled and ks.include_in_trade_score) else 0.0)
    b_tot = b_core + (ks.keeper_weight*kdb if (ks.enabled and ks.include_in_trade_score) else 0.0)

    res = {
        "mode": mode,
        "value_col": value_col,
        "team_a_delta": a_tot,
        "team_b_delta": b_tot,
        "total_fairness": a_tot + b_tot,
        "team_a_delta_core": a_core,
        "team_b_delta_core": b_core,
        "keeper_delta_team_a": kda,
        "keeper_delta_team_b": kdb,
        "team_a_in_breakdown": a_in[["name","pos",value_col,"keeper_surplus","keeper_eligible"] if ks.enabled else ["name","pos",value_col]].rename(columns={value_col:"value"}).to_dict(orient="records"),
        "team_a_out_breakdown": a_out[["name","pos",value_col,"keeper_surplus","keeper_eligible"] if ks.enabled else ["name","pos",value_col]].rename(columns={value_col:"value"}).to_dict(orient="records"),
        "team_b_in_breakdown": b_in[["name","pos",value_col,"keeper_surplus","keeper_eligible"] if ks.enabled else ["name","pos",value_col]].rename(columns={value_col:"value"}).to_dict(orient="records"),
        "team_b_out_breakdown": b_out[["name","pos",value_col,"keeper_surplus","keeper_eligible"] if ks.enabled else ["name","pos",value_col]].rename(columns={value_col:"value"}).to_dict(orient="records"),
    }
    res.update(picks_detail)
    return res
