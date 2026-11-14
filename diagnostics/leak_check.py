import argparse
import numpy as np
import Loading_Data
import protocol_mine as P


def emotion_of(trials):
    return P.trial_to_emotion().numpy()[trials]


def knn1(Xtr, ytr, Xte, bs=512):
    out = []
    for i in range(0, len(Xte), bs):
        d = ((Xte[i:i + bs, None, :] - Xtr[None]) ** 2).sum(-1)
        out.append(ytr[d.argmin(1)])
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--test_frac', type=float, default=0.3)
    ap.add_argument('--gap', type=int, default=10)
    a = ap.parse_args()

    res = {'base': [], 'fair': []}
    for t in a.targets:
        data, trials = Loading_Data.get_data_label(str(t))
        X = data.reshape(len(data), -1).astype(np.float32)
        trials = np.asarray(trials).astype(int)
        row = [f'target {t:2d}']
        for proto in ('base', 'fair'):
            lab, _, tst = P.split_by_protocol(trials, a.n, a.test_frac, a.gap, proto)
            pred = knn1(X[lab], trials[lab], X[tst])  # predicts trial id
            acc = (emotion_of(pred) == emotion_of(trials[tst])).mean()
            overlap = len(set(lab) & set(tst)) / len(tst)
            res[proto].append(acc)
            row.append(f'{proto}: acc={acc:.3f} (labelled-in-test={overlap:.1%})')
        print(' | '.join(row))
    print('\nMEAN  base=%.3f  fair=%.3f  chance=0.333' % (np.mean(res['base']), np.mean(res['fair'])))
    print('gap (leak size) = %.3f' % (np.mean(res['base']) - np.mean(res['fair'])))


if __name__ == '__main__':
    main()
