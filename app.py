
# --- robust import path for Streamlit Cloud ---
import os, sys
PARENT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
ROOT = os.path.dirname(PARENT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ----------------------------------------------

import re, math, requests
from pathlib import Path
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd

from tradeanalyzer.valuation import (
    LeagueSettings, ScoringSettings, DynastySettings, KeeperSettings,
    apply_scoring, vorp_by_position, dynasty_values, apply_keeper_model,
    trade_value, Trade
)

st.set_page_config(page_title="FF Trade Analyzer", page_icon="🏈", layout="wide")
st.title("🏈 Fantasy Football Trade Analyzer")

# League defaults (12-team, QB/RB/RB/WR/WR/TE/2×FLEX/DEF)
scoring = ScoringSettings()
roster_starters = {"QB":1, "RB":2, "WR":2, "TE":1, "FLEX":2, "SUPERFLEX":0, "DEF":1, "K":0}
ls = LeagueSettings(teams=12, roster_starters=roster_starters, scoring=scoring)

TEAM_PAREN_RE = re.compile(r"\s*\([A-Z]{2,4}\)$")
POS_PAREN_RE = re.compile(r"\s*\((QB|RB|WR|TE|DEF|DST)\)$", re.I)

def clean_name(name: str) -> str:
    s = str(name).strip()
    s = POS_PAREN_RE.sub('', s)
    s = TEAM_PAREN_RE.sub('', s)
    s = re.sub(r"\s+", " ", s)
    return s

# -------------------- Rankings helpers --------------------

def ranks_to_values(df_ranks: pd.DataFrame, rank_col: str, name_col: str, pos_col: str = None) -> pd.DataFrame:
    df = df_ranks.copy()
    df = df[df[rank_col].notna()].copy()
    df[rank_col] = df[rank_col].astype(int)
    df["name"] = df[name_col].astype(str).apply(clean_name)
    if pos_col and pos_col in df.columns:
        df["pos"] = df[pos_col].astype(str).str.upper().str.replace("DST","DEF", regex=False)
    else:
        pat = re.compile(r"(QB|RB|WR|TE|DEF|DST)", re.I)
        df["pos"] = df["name"].str.extract(pat, expand=False).str.upper().str.replace("DST","DEF", regex=False)
    df["value_redraft"] = 120 - df[rank_col]
    out = df[["name","pos","value_redraft"]].dropna(subset=["name"]).copy()
    out["pos"] = out["pos"].fillna("WR")
    out["draft_round"] = math.nan
    out["years_kept"] = 0
    out["risk"] = 1.0
    out["proj_pts_raw"] = 0.0
    out["vorp_adj"] = 0.0
    return out

@st.cache_data(show_spinner=False)
def fetch_espn_ros_top100() -> pd.DataFrame:
    url = "https://www.espn.com/fantasy/football/story/_/id/46205637/fantasy-football-2025-updated-rest-season-rankings"
    r = requests.get(url, timeout=25); r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    text = soup.get_text("
", strip=False)
    block, capturing = [], False
    for line in text.splitlines():
        if "Overall top 100" in line:
            capturing = True
            continue
        if capturing:
            if "Quarterback" in line or "Top 40 Quarterbacks" in line:
                break
            if re.match(r"^\s*\d+[\.|\s]", line):
                block.append(line.strip())
    rows = []
    for s in block:
        m = re.match(r"^\s*(\d+)[\.|\)]?\s+(.*)$", s)
        if not m: 
            continue
        rows.append({"overall": int(m.group(1)), "player": m.group(2).strip()})
    if not rows:
        raise RuntimeError("Could not locate ESPN 'Overall top 100' block.")
    df = pd.DataFrame(rows)
    return ranks_to_values(df, rank_col="overall", name_col="player")

@st.cache_data(show_spinner=False)
def fetch_fantasypros_ros_ppr() -> pd.DataFrame:
    url = "https://www.fantasypros.com/nfl/rankings/?type=ros&scoring=PPR"
    tables = pd.read_html(url)
    if not tables:
        raise RuntimeError("No tables found on FantasyPros page.")
    df0 = tables[0].copy()
    rank_col = [c for c in df0.columns if str(c).strip().lower().startswith("rank")][0]
    name_col = [c for c in df0.columns if "Player" in str(c)][0]
    df0.rename(columns={rank_col: "overall", name_col: "player"}, inplace=True)
    return ranks_to_values(df0, rank_col="overall", name_col="player")

@st.cache_data(show_spinner=False)
def fetch_draftsharks_ros_ppr() -> pd.DataFrame:
    url = "https://www.draftsharks.com/ros-rankings/ppr"
    tables = pd.read_html(url)
    if not tables:
        raise RuntimeError("No tables found on DraftSharks ROS page.")
    df0 = tables[0].copy()
    candidates = [c for c in df0.columns if str(c).strip().lower() in ("rk","rank","overall","ovr","#")]
    if not candidates:
        candidates = [df0.columns[0]]
    rank_col = candidates[0]
    name_col = [c for c in df0.columns if str(c).strip().lower().startswith('player')]
    if not name_col:
        name_col = [df0.columns[1]]
    name_col = name_col[0]
    def ds_name(x: str) -> str:
        s = str(x)
        m = re.match(r"^([A-Z][a-zA-Z'\.\-]+\s+[A-Z][a-zA-Z'\.\-]+)", s)
        return m.group(1) if m else s.split('  ')[0].strip()
    df0['player'] = df0[name_col].apply(ds_name)
    df0.rename(columns={rank_col:'overall'}, inplace=True)
    df0 = df0[['overall','player']]
    return ranks_to_values(df0, rank_col='overall', name_col='player')

with st.sidebar:
    st.header("Mode & Models")
    dyn_enabled = st.toggle("Dynasty mode", value=False)
    window_years = st.slider("Dynasty window (years)", 1, 5, 3, disabled=not dyn_enabled)
    discount = st.slider("Discount rate", 0.00, 0.30, 0.12, 0.01, disabled=not dyn_enabled)
    bias = st.slider("Lifecycle bias (Rebuild -1 ↔ +1 Win Now)", -1.0, 1.0, 0.0, 0.1, disabled=not dyn_enabled)

    st.divider()
    st.header("Keeper Settings")
    keeper_enabled = st.toggle("Enable keeper model", value=True)
    st.caption("R1/R2 ineligible. Next-year cost = round − 2. Keep-4 cap applies.")
    include_keeper = st.toggle("Include keeper surplus in trade score", value=True, disabled=not keeper_enabled)
    keep_slots = st.number_input("Keeper cap per team", 0, 10, 4, disabled=not keeper_enabled)

    st.divider()
    st.header("Data Source")
    data_source = st.radio(
        "Where do player values come from?",
        ["Upload CSV (projections/stats)", "Website rankings (ROS)"], index=1
    )
    provider = None
    run_fetch = False
    proj_file = None
    if data_source == "Website rankings (ROS)":
        provider = st.selectbox("Ranking provider", [
            "Local snapshot (GitHub Actions)",
            "ESPN ROS (Overall Top 100)",
            "FantasyPros ROS (PPR Overall)",
            "DraftSharks ROS (PPR)"
        ])
        run_fetch = st.button("Load rankings", type="primary")
    else:
        proj_file = st.file_uploader("Upload projections CSV", type=["csv"]) 
        st.caption("Minimum columns: name, pos. Others optional.")

# Build model settings
ds = DynastySettings(enabled=dyn_enabled, window_years=window_years, discount_rate=discount, win_now_bias=bias)
ks = KeeperSettings(enabled=keeper_enabled, include_in_trade_score=include_keeper,
                    keeper_slots_team_a=keep_slots, keeper_slots_team_b=keep_slots,
                    min_keep_round=3, disallowed_rounds=[1,2])

st.header("Build Player Values")

df = None
if data_source == "Upload CSV (projections/stats)":
    if proj_file is None:
        st.info("Upload a projections CSV (or switch to Website rankings).")
    else:
        df_raw = pd.read_csv(proj_file)
        if "name" not in df_raw.columns or "pos" not in df_raw.columns:
            st.error("Missing required columns: name and/or pos")
            st.stop()
        df = apply_scoring(df_raw, ls)
        df = vorp_by_position(df, ls)
        df = dynasty_values(df, ls, ds)
        df = apply_keeper_model(df, ls, ds, ks)
else:
    st.info("Rankings mode converts site ranks → values. No projections needed.")
    if run_fetch:
        try:
            if provider.startswith("Local snapshot"):
                snap_path = Path(__file__).resolve().parent / 'data' / 'ros_snapshot.csv'
                if not snap_path.exists():
                    st.error("No local snapshot found. Wait for the daily GitHub Action or run the fetch script locally.")
                    st.stop()
                df = pd.read_csv(snap_path)
                # scaffold minimal columns
                for col,default in [("draft_round",math.nan),("years_kept",0),("risk",1.0),("proj_pts_raw",0.0),("vorp_adj",0.0)]:
                    if col not in df.columns: df[col] = default
            elif provider.startswith("ESPN"):
                df = fetch_espn_ros_top100()
            elif provider.startswith("FantasyPros"):
                df = fetch_fantasypros_ros_ppr()
            elif provider.startswith("DraftSharks"):
                df = fetch_draftsharks_ros_ppr()
            else:
                st.error("Unsupported provider selection")
                st.stop()
            df = dynasty_values(df, ls, ds)
            df = apply_keeper_model(df, ls, ds, ks)
            st.success(f"Loaded {len(df)} players from {provider}")
            with st.expander("Preview values (rankings)", expanded=False):
                st.dataframe(df[["name","pos","value_redraft","keeper_eligible","keeper_surplus"]], use_container_width=True)
        except Exception as e:
            st.error(f"Fetch/parse error: {e}")
            st.stop()

if df is None:
    st.stop()

# -------------------- Compare helpers (UI-only) --------------------

def _pick_curve(superflex: bool = False):
    r1 = [60,56,53,50,48,46,44,42,40,38,36,34]
    r2 = [28,27,26,25,24,23,22,21,20,19,18,17]
    r3 = [12,11,10,9,9,8,8,7,7,6,6,5]
    r4 = [4,4,3,3,3,2.5,2.5,2,2,1.5,1.5,1.0]
    curve = {(1,i+1):v for i,v in enumerate(r1)}
    curve.update({(2,i+1):v for i,v in enumerate(r2)})
    curve.update({(3,i+1):v for i,v in enumerate(r3)})
    curve.update({(4,i+1):v for i,v in enumerate(r4)})
    if superflex:
        for sel in range(1,7): curve[(1,sel)] *= 1.25
        for sel in range(7,13): curve[(1,sel)] *= 1.10
        for sel in range(1,5): curve[(2,sel)] *= 1.05
    return curve

def _verdict(a_tot: float, b_tot: float, eps: float = 1.0) -> str:
    if abs(a_tot - b_tot) <= eps:
        return "Even"
    return "Team A" if a_tot > b_tot else "Team B"

def _balance_suggestion(a_tot: float, b_tot: float, superflex: bool = False, target_year: int = None):
    import pandas as pd
    if target_year is None:
        target_year = pd.Timestamp.today().year
    if a_tot >= b_tot:
        winner = "Team A"; needed = max(a_tot, 0.0)
    else:
        winner = "Team B"; needed = max(b_tot, 0.0)
    if needed < 0.75:
        return winner, None, None
    curve = _pick_curve(superflex)
    (rnd, sel), val = min(curve.items(), key=lambda kv: abs(kv[1]-needed))
    pick_str = f"{target_year} {rnd}.{sel:02d}"
    return winner, pick_str, val


def render_compare_summary(st, result: dict, ls):
    a_tot = float(result.get("team_a_delta", 0.0))
    b_tot = float(result.get("team_b_delta", 0.0))
    fairness = float(result.get("total_fairness", 0.0))
    kda = float(result.get("keeper_delta_team_a", 0.0))
    kdb = float(result.get("keeper_delta_team_b", 0.0))

    verdict = _verdict(a_tot, b_tot)
    st.subheader("Compare Result")
    st.write(
        f"**Verdict:** {verdict} — "
        f"A Δ **{a_tot:+.2f}**, B Δ **{b_tot:+.2f}**, Total **{fairness:+.2f}**"
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Team A Δ (incl. Keeper)", f"{a_tot:+.2f}")
        st.caption(f"Keeper impact: {kda:+.2f}")
    with m2:
        st.metric("Team B Δ (incl. Keeper)", f"{b_tot:+.2f}")
        st.caption(f"Keeper impact: {kdb:+.2f}")
    with m3:
        st.metric("Total Fairness", f"{fairness:+.2f}")
        st.caption("0.00 → perfectly even under this model")

    winner, pick_str, val = _balance_suggestion(a_tot, b_tot, ls.superflex)
    st.divider()
    st.subheader("How to balance it (suggestion)")
    if pick_str is None:
        st.info("This looks close enough—no pick suggestion needed.")
    else:
        if winner == "Team A":
            st.write(f"Ask **Team A** to add **{pick_str}** (≈ **{val:.1f}** value).")
        else:
            st.write(f"Ask **Team B** to add **{pick_str}** (≈ **{val:.1f}** value).")

    with st.expander("What these numbers mean"):
        st.markdown(
            "- **Δ (delta)** is net value change for that team.
"
            "- **Total Fairness** is A Δ + B Δ (can be non‑zero due to roster/keeper effects).
"
            "- **Pick suggestion** uses the same pick curve & Superflex tilt as your engine."
        )

# -------------------- Trade UI --------------------
st.header("Define Trade")
names = df["name"].tolist()
c1, c2 = st.columns(2)
with c1:
    a_out = st.multiselect("Team A sends (players)", names, key="a_out")
    a_picks = st.text_input("Team A sends picks (comma separated)", value="")
with c2:
    b_out = st.multiselect("Team B sends (players)", names, key="b_out")
    b_picks = st.text_input("Team B sends picks (comma separated)", value="")

st.caption("Pick format: 2026 1.03, 2026 2.05, or '2026 3' (round only)")

def parse_picks(s: str):
    return [t.strip() for t in s.split(',') if t.strip()]

r1, r2 = st.columns(2)
with r1:
    a_roster = {"QB":st.number_input("A: QB filled",0,3,0),"RB":st.number_input("A: RB filled",0,6,0),
                "WR":st.number_input("A: WR filled",0,8,0),"TE":st.number_input("A: TE filled",0,3,0),
                "DEF":st.number_input("A: DEF filled",0,3,0),"K":0}
with r2:
    b_roster = {"QB":st.number_input("B: QB filled",0,3,0),"RB":st.number_input("B: RB filled",0,6,0),
                "WR":st.number_input("B: WR filled",0,8,0),"TE":st.number_input("B: TE filled",0,3,0),
                "DEF":st.number_input("B: DEF filled",0,3,0),"K":0}

# --- Compare vs Full Analyze buttons ---
btn_left, btn_right = st.columns([1,1])
with btn_left:
    compare_btn = st.button("Compare (A vs B)", help="Quick verdict & summary", key="btn_compare")
with btn_right:
    analyze_btn = st.button("Full Analyze", type="primary", help="Detailed breakdown tables", key="btn_full")

if compare_btn or analyze_btn:
    trade = Trade(team_a_gives=a_out, team_b_gives=b_out,
                  team_a_picks_gives=parse_picks(a_picks), team_b_picks_gives=parse_picks(b_picks))
    result = trade_value(df, trade, a_roster, b_roster, ls, ds, ks, mode=("dynasty" if ds.enabled else "redraft"))
    if "error" in result:
        st.error(result["error"]) 
    else:
        if compare_btn:
            render_compare_summary(st, result, ls)
        else:
            st.success(f"Team A Δ: {result['team_a_delta']:+.2f} | Team B Δ: {result['team_b_delta']:+.2f} | Total: {result['total_fairness']:+.2f}")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Team A Incoming")
                st.dataframe(pd.DataFrame(result["team_a_in_breakdown"]), use_container_width=True)
                st.subheader("Team A Outgoing")
                st.dataframe(pd.DataFrame(result["team_a_out_breakdown"]), use_container_width=True)
            with c2:
                st.subheader("Team B Incoming")
                st.dataframe(pd.DataFrame(result["team_b_in_breakdown"]), use_container_width=True)
                st.subheader("Team B Outgoing")
                st.dataframe(pd.DataFrame(result["team_b_out_breakdown"]), use_container_width=True)
