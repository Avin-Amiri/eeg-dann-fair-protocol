import csv, sys
import numpy as np

rows = list(csv.DictReader(open(sys.argv[1])))
print("rows in file:", len(rows), "(full 15 targets = 210)")
tg = sorted({int(r['target']) for r in rows})
M, P, B = [], [], []
for t in tg:
    x = [r for r in rows if int(r['target']) == t]
    a = np.array([float(r['acc_final']) for r in x])
    c = np.array([float(r['cal_loss_final']) for r in x])
    pick = a[np.argmin(c)]
    M.append(a.mean()); P.append(pick); B.append(a.max())
    print("target %2d  pairs=%2d  mean=%.3f  picked_by_calibration=%.3f  best(oracle)=%.3f" % (t, len(x), a.mean(), pick, a.max()))
print("AVERAGE   mean=%.3f  picked=%.3f  oracle=%.3f" % (np.mean(M), np.mean(P), np.mean(B)))
