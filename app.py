
# --- robust import path for Streamlit Cloud ---
import os, sys
PARENT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
ROOT = os.path.dirname(PARENT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ----------------------------------------------

import re, math, requests
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

# -------------------- Rankings helpers --------------------
def ranks_to_values(df_ranks: pd.DataFrame, rank_col: str, name_col: str, pos_col: str = None) -> pd.DataFrame:
    df = df_ranks.copy()
    df = df[df[rank_col].notna()].copy()
    df[rank_col] = df[rank_col].astype(int)
    df["name"] = df[name_col].astype(str).str.strip()
    if pos_col and pos_col in df.columns:
        df["pos"] = df[pos_col].astype(str).str.upper().str.replace("DST","DEF", regex=False)
    else:
        pat = re.compile(r"\((QB|RB|WR|TE|DEF|DST)\)$", re.I)
        def infer_pos(nm):
            m = pat.search(nm)
            return (m.group(1).upper().replace("DST","DEF") if m else None)
        df["pos"] = df["name"].apply(infer_pos)
        df["name"] = df["name"].str.replace(r"\s*\((QB|RB|WR|TE|DEF|DST)\)$", "", regex=True)
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
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    from bs4 import BeautifulSoup as _BS
    soup = _BS(r.text, "lxml")
    text = soup.get_text("
", strip=False)
    block = []
    capturing = False
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
        provider = st.selectbox("Ranking provider", ["ESPN ROS (Overall Top 100)", "FantasyPros ROS (PPR Overall)"])
        run_fetch = st.button("Fetch rankings", type="primary")
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
            if provider == "ESPN ROS (Overall Top 100)":
                df = fetch_espn_ros_top100()
            elif provider == "FantasyPros ROS (PPR Overall)":
                df = fetch_fantasypros_ros_ppr()
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

if st.button("Analyze Trade", type="primary"):
    trade = Trade(team_a_gives=a_out, team_b_gives=b_out,
                  team_a_picks_gives=parse_picks(a_picks), team_b_picks_gives=parse_picks(b_picks))
    result = trade_value(df, trade, a_roster, b_roster, ls, ds, ks, mode=("dynasty" if ds.enabled else "redraft"))
    if "error" in result:
        st.error(result["error"]) 
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
