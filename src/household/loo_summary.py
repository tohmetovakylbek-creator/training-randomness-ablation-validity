import json, glob, os

rows = []
for p in glob.glob("results_norm/loo/*/meta_*_y_final_components.json"):
    j = json.load(open(p, encoding="utf-8"))
    u = os.path.basename(os.path.dirname(p))
    m, rk = j["model"], j["required_k"]
    rows.append((u, m["mu"] * 100, m["I2"], rk.get("0.01"), rk.get("0.02")))

print(len(rows), "прогонов")
print("объект".ljust(20), "mu,%".rjust(8), "I2,%".rjust(7), "k(1%)".rjust(7), "k(2%)".rjust(7))
for u, mu, i2, k1, k2 in sorted(rows, key=lambda r: (r[4] is None, r[4])):
    print(u.ljust(20), ("%+.3f" % mu).rjust(8), ("%.1f" % i2).rjust(7),
          str(k1).rjust(7), str(k2).rjust(7))