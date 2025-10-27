
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import pandas as pd, numpy as np

@dataclass
class ScoringSettings: pass

@dataclass
class LeagueSettings:
    teams: int
    roster_starters: Dict[str,int]
    ppr: float = 1.0
    te_premium: float = 0.0
    qb_passing_td: int = 4
    superflex: bool = False
    scoring: ScoringSettings = field(default_factory=ScoringSettings)

@dataclass
class DynastySettings:
    enabled: bool = False
    window_years: int = 3
    discount_rate: float = 0.12
    win_now_bias: float = 0.0

@dataclass
class KeeperSettings:
    enabled: bool = False
    include_in_trade_score: bool = True
    keeper_slots_team_a: Optional[int] = 4
    keeper_slots_team_b: Optional[int] = 4
    min_keep_round: int = 3
    disallowed_rounds: Optional[List[int]] = None

@dataclass
class Trade:
    team_a_gives: List[str]
    team_b_gives: List[str]
    team_a_picks_gives: Optional[List[str]] = None
    team_b_picks_gives: Optional[List[str]] = None

# Minimal stubs to keep app functional in this sandbox (values come from rankings directly)
def apply_scoring(df, ls): return df

def vorp_by_position(df, ls): return df

def dynasty_values(df, ls, ds):
    if not ds.enabled:
        df = df.copy(); df['value_dynasty'] = df['value_redraft']; return df
    df = df.copy(); df['value_dynasty'] = df['value_redraft'] * 0.95
    return df


def apply_keeper_model(df, ls, ds, ks):
    df = df.copy()
    df['keeper_eligible'] = True
    df['keeper_surplus'] = df['value_redraft']*0.05
    return df

# Simple pick curve
_DEF_CURVE = {**{(1,i+1):v for i,v in enumerate([60,56,53,50,48,46,44,42,40,38,36,34])},
              **{(2,i+1):v for i,v in enumerate([28,27,26,25,24,23,22,21,20,19,18,17])},
              **{(3,i+1):v for i,v in enumerate([12,11,10,9,9,8,8,7,7,6,6,5])},
              **{(4,i+1):v for i,v in enumerate([4,4,3,3,3,2.5,2.5,2,2,1.5,1.5,1.0])}}

def _parse_pick(pick:str):
    s = pick.upper().replace('-', ' ').replace('.', ' ')
    parts = [p for p in s.split() if p]
    year = int(parts[0]); rnd = int(parts[1]); sel = int(parts[2]) if len(parts)>2 else 1
    return year, rnd, sel

def value_future_pick(pick: str, ls: LeagueSettings, ds: DynastySettings, current_year: int=None) -> float:
    _, rnd, sel = _parse_pick(pick)
    base = _DEF_CURVE.get((rnd, sel), 0.5)
    return base

# trade_value using redraft/dynasty value columns

def trade_value(df_values, trade: Trade, a_counts, b_counts, ls, ds, ks, mode='redraft'):
    value_col = 'value_dynasty' if (ds.enabled and mode=='dynasty') else 'value_redraft'
    df = df_values.set_index('name', drop=False)
    def sumv(names):
        if not names: return 0.0, []
        sub = df.loc[names]
        return float(sub[value_col].sum()), sub[["name","pos",value_col]].rename(columns={value_col:"value"}).to_dict('records')
    a_out_v, a_out_rows = sumv(trade.team_a_gives)
    b_out_v, b_out_rows = sumv(trade.team_b_gives)
    a_in_v, a_in_rows = b_out_v, b_out_rows
    b_in_v, b_in_rows = a_out_v, a_out_rows

    # picks (dynasty mode only in the full build; included here for structure)
    a_out_picks = trade.team_a_picks_gives or []
    b_out_picks = trade.team_b_picks_gives or []
    a_out_pv = sum(value_future_pick(p, ls, ds) for p in a_out_picks)
    b_out_pv = sum(value_future_pick(p, ls, ds) for p in b_out_picks)
    a_in_v += b_out_pv; a_out_v += a_out_pv
    b_in_v += a_out_pv; b_out_v += b_out_pv

    a_core = a_in_v - a_out_v
    b_core = b_in_v - b_out_v

    # keeper deltas (simple placeholder)
    kda = 0.0; kdb = 0.0
    res = {
        'team_a_delta': a_core,
        'team_b_delta': b_core,
        'total_fairness': a_core + b_core,
        'keeper_delta_team_a': kda,
        'keeper_delta_team_b': kdb,
        'team_a_in_breakdown': a_in_rows,
        'team_a_out_breakdown': a_out_rows,
        'team_b_in_breakdown': b_in_rows,
        'team_b_out_breakdown': b_out_rows,
    }
    return res
