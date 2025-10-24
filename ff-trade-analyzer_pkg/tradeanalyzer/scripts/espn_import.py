
"""
ESPN importer: build a roster CSV from a league (requires SWID / ESPN_S2 for private leagues).
Usage examples:
  python tradeanalyzer/scripts/espn_import.py --league 639585 --season 2025 --public
  python tradeanalyzer/scripts/espn_import.py --league 639585 --season 2025 --swid "{...}" --espn_s2 "..."
"""
import argparse
from pathlib import Path
import requests
import pandas as pd

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{league}"
VIEWS = ["mTeam","mRoster","mSettings"]
POS_MAP = {1:"QB",2:"RB",3:"WR",4:"TE",5:"K",16:"DEF"}

def fetch(season:int, league:int, swid:str=None, espn_s2:str=None, public:bool=False):
    url = BASE.format(season=season, league=league)
    params = [("view", v) for v in VIEWS]
    cookies = None if public else {"SWID": swid, "espn_s2": espn_s2}
    r = requests.get(url, params=params, cookies=cookies, timeout=30)
    if r.status_code == 401:
        raise SystemExit("401 Unauthorized. League is private or cookies invalid.")
    r.raise_for_status()
    return r.json()

def build(payload: dict) -> pd.DataFrame:
    rows = []
    for t in payload.get("teams", []):
        for e in (t.get("roster") or {}).get("entries", []):
            p = (e.get("playerPoolEntry") or {}).get("player") or {}
            name = p.get("fullName") or p.get("name") or ""
            pos = POS_MAP.get(p.get("defaultPositionId"), None)
            if p.get("defaultPositionId") == 16 or "D/ST" in (name or '').upper():
                pos = "DEF"
            if not name or not pos:
                continue
            rows.append({"name": name, "pos": pos, "age": p.get("age", ""), "risk": 1.0})
    df = pd.DataFrame(rows).drop_duplicates(subset=["name","pos"]).reset_index(drop=True)
    # add zeros for stat columns your scoring pipeline expects
    for c in [
        'proj_pass_yd','proj_pass_td','proj_pass_td50','proj_int','proj_pass_2pt','proj_400p_games',
        'proj_rush_yd','proj_rush_td','proj_rush_td50','proj_rush_2pt','proj_100r_games','proj_200r_games',
        'proj_rec','proj_rec_yd','proj_rec_td','proj_rec_td50','proj_rec_2pt','proj_100re_games','proj_200re_games'
    ]:
        if c not in df.columns:
            df[c] = 0
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", type=int, required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--swid", type=str)
    ap.add_argument("--espn_s2", type=str)
    ap.add_argument("--out", type=str, default="espn_league_export.csv")
    args = ap.parse_args()

    payload = fetch(args.season, args.league, args.swid, args.espn_s2, args.public)
    df = build(payload)
    Path(args.out).write_text(df.to_csv(index=False))
    print(f"Wrote {len(df)} players → {args.out}")

if __name__ == "__main__":
    main()
