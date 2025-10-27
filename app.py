# --- robust import path for Streamlit Cloud ---
import os, sys
PARENT = os.path.dirname(os.path.abspath(__file__))  # /mount/src/tradeanalyzer
ROOT = os.path.dirname(PARENT)                       # /mount/src
if ROOT not in sys.path:
   sys.path.insert(0, ROOT)
# ----------------------------------------------
import streamlit as st
import pandas as pd
# Use absolute package import so it works when Streamlit runs the file as a script
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
    st.header("Upload Projections CSV")
    proj_file = st.file_uploader("CSV with projections", type=["csv"]) 
    st.caption("Try the sample in tradeanalyzer/data/sample_projections.csv")

# Build settings

ds = DynastySettings(enabled=dyn_enabled, window_years=window_years, discount_rate=discount, win_now_bias=bias)
ks = KeeperSettings(enabled=keeper_enabled, include_in_trade_score=include_keeper,
                    keeper_slots_team_a=keep_slots, keeper_slots_team_b=keep_slots,
                    min_keep_round=3, disallowed_rounds=[1,2])

st.header("Upload & Score Players")
if proj_file is None:
    st.info("Upload a projections CSV to begin (or use the sample CSV).")
else:
    df_raw = pd.read_csv(proj_file)
    for col in ["name","pos"]:
        if col not in df_raw.columns:
            st.error(f"Missing required column: {col}")
            st.stop()

    # Pipeline: scoring → VORP → dynasty → keepers
    df = apply_scoring(df_raw, ls)
    df = vorp_by_position(df, ls)
    df = dynasty_values(df, ls, ds)
    df = apply_keeper_model(df, ls, ds, ks)

    value_col = "value_dynasty" if ds.enabled else "value_redraft"

    with st.expander("Values table", expanded=False):
        cols = ["name","pos","proj_pts_raw","replacement_pts","vorp","value_redraft"]
        if ds.enabled: cols.append("value_dynasty")
        if keeper_enabled:
            cols += ["draft_round","years_kept","keeper_eligible","keeper_next_round",
                     "keeper_expected_value_next_round","keeper_surplus"]
        st.dataframe(df[cols], use_container_width=True)

    st.header("Define Trade")
    names = df["name"].tolist()
    c1, c2 = st.columns(2)
    with c1:
        a_out = st.multiselect("Team A sends (players)", names, key="a_out")
        a_picks = st.text_input("Team A sends picks (comma separated)", value="")
    with c2:
        b_out = st.multiselect("Team B sends (players)", names, key="b_out")
        b_picks = st.text_input("Team B sends picks (comma separated)", value="")

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
