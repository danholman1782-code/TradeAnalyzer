# Fantasy Football Trade Analyzer (Streamlit)

This repo contains a Streamlit app for analyzing fantasy football trades with:

- Custom **PPR** scoring (bonuses included)
- **2×FLEX + DEF** lineup and VORP (replacement) math
- Optional **Dynasty** window (multi‑year PV with age/decay curves)
- **Keepers**: **R1/R2 ineligible**, next‑year cost = **round − 2** (min R1), **Keep‑4** cap

## Run locally
```bash
pip install -r requirements.txt
streamlit run tradeanalyzer/app.py
```

Upload `tradeanalyzer/data/sample_projections.csv` (or your projections) then define a trade in the UI.

## Streamlit Cloud deploy
1. Push this repo to **public GitHub**.
2. Open https://streamlit.io/cloud → **New app** → choose repo → main file: `tradeanalyzer/app.py`.
3. (Optional) Set **Secrets** for ESPN import in App → Settings → Secrets.

## Secrets (optional, ESPN import)
Do **not** commit real secrets. Use `tradeanalyzer/.streamlit/secrets.toml` (gitignored) locally and the Secrets panel in Streamlit Cloud.
```toml
[espn]
SWID = "{YOUR_SWID}"
ESPN_S2 = "YOUR_ESPN_S2"
```

---

### CSV schema (season-level expectations)
Required: `name,pos`

Optional scoring columns (missing values default to 0):
- Passing: `proj_pass_yd, proj_pass_td, proj_pass_td50, proj_int, proj_pass_2pt, proj_400p_games`
- Rushing: `proj_rush_yd, proj_rush_td, proj_rush_td50, proj_rush_2pt, proj_100r_games, proj_200r_games`
- Receiving: `proj_rec, proj_rec_yd, proj_rec_td, proj_rec_td50, proj_rec_2pt, proj_100re_games, proj_200re_games`
- Defense (pos=DEF/DST): category counts & PA/YA buckets

Dynasty/Keeper helpers (optional): `age, risk, draft_round, years_kept`
