"""Payoff breakdown for the winning strategy: 2x cross, top-5% (and top-10%) model score.
Fire rule, trades/day, profit across bet sizes, and EV sensitivity to entry slip (the
slot/latency axis). All gap-0-assumed; realized = these x P(gap-0). OOS test set."""
import json, numpy as np, time
from sklearn.ensemble import HistGradientBoostingClassifier

PANEL = "/root/the-distribution-will-manifest/bot_data/cont_multi_panel.jsonl"
TIP_SOL = 0.005          # fixed Jito tip per attempt
PUMP_RT = 0.02           # ~1% pump fee each side, round trip
cross = {}; outc = {}; fillm = {}
for l in open(PANEL):
    try: e = json.loads(l)
    except Exception: continue
    k = e.get("kind"); key = (e.get("mint"), e.get("mult"))
    if k == "cross": cross[key] = e
    elif k == "outcome": outc[key] = e
    elif k == "fill": fillm[key] = e
rows = []
for key, o in outc.items():
    c = cross.get(key)
    if c is None or key[1] != 2.0: continue          # 2x only
    f = fillm.get(key)
    slip = (f["fill_mid"] / c["cross_mid"] - 1.0) if (f and c.get("cross_mid")) else 0.0
    rows.append([time.strftime("%m-%d", time.gmtime(c["t"])), c["dd"], c["buy_frac"], c["ntr"],
                 c["recent"], c.get("tps", 0), c.get("uniq", 0), o["y"], o["ret"], slip])
day = np.array([r[0] for r in rows]); X = np.array([[r[1],r[2],r[3],r[4],r[5],r[6]] for r in rows], float)
y = np.array([r[7] for r in rows], float); ret = np.array([r[8] for r in rows], float); slip = np.array([r[9] for r in rows], float)
days = sorted(set(day)); trd = set(days[:3]); tr = np.array([d in trd for d in day]); te = ~tr
clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05).fit(X[tr], y[tr])
p = clf.predict_proba(X[te])[:, 1]
ret_te = ret[te]; y_te = y[te]; slip_te = slip[te]
n_test_days = len(set(day[te]))
print(f"2x crosses: train={int(tr.sum())} test={int(te.sum())} over {n_test_days} test days\n")

for q, lbl in [(0.95, "TOP-5%"), (0.90, "TOP-10%")]:
    cut = np.quantile(p, q); s = p >= cut
    n = int(s.sum()); per_day = n / n_test_days
    mid = ret_te[s].mean(); hit = y_te[s].mean(); slip_med = np.median(slip_te[s])
    print(f"=== {lbl}: fire when model p >= {cut:.3f}  |  {n} trades / {n_test_days}d = {per_day:.0f}/day  |  "
          f"hit {hit:.0%}  gap-0 mid_ret {mid:+.3f}  median entry-slip {slip_med:.1%} ===")
    print(f"  {'bet':>6} {'net/trade':>10} {'/day gap0':>10} {'/day xP=.35':>12} {'/day xP=.50':>12}")
    for bet in [0.05, 0.1, 0.25, 0.5, 1.0]:
        net_ret = mid - PUMP_RT - TIP_SOL / bet          # honest per-trade return on the bet
        net_sol = bet * net_ret                          # SOL per filled gap-0 trade
        print(f"  {bet:>6.2f} {net_sol:>+10.4f} {net_sol*per_day:>+10.2f} "
              f"{net_sol*per_day*0.35:>+12.2f} {net_sol*per_day*0.50:>+12.2f}")
    # EV sensitivity to entry slip (the slot/latency axis): filling s% higher cuts return ~s
    print(f"  slip/slot sensitivity (net_ret at 0.5 bet; cap reverts beyond ~25%):")
    for sl in [0.0, 0.05, 0.10, 0.20]:
        nr = (mid - sl) - PUMP_RT - TIP_SOL / 0.5
        print(f"    +{sl:.0%} slip (~{'gap-0' if sl==0 else 'gap-1' if sl<=0.1 else 'gap-2+'}): net_ret {nr:+.3f}")
    print()
print("realized = these x P(gap-0); breakeven P(gap-0) ~20%. Beyond ~25% slip the buy reverts (cheap miss).")
