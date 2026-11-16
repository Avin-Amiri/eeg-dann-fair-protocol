import argparse
import csv
import json
import os

import numpy as np
import torch

from data import Loading_Data
from protocols import protocol_mine
from models.main_models_seed_6 import Classifier, Encoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--stage', type=str, default='final', choices=['final', 'step3'])
    p.add_argument('--top_k', type=int, default=5)
    return p.parse_args()


def read_results(run_root):
    rows = []
    with open(os.path.join(run_root, 'results.csv'), newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


@torch.no_grad()
def predict_emotion(path, X, device, mapping):
    ckpt = torch.load(path, map_location=device)
    encoder = Encoder().to(device)
    classifier = Classifier().to(device)
    encoder.load_state_dict(ckpt['encoder'])
    classifier.load_state_dict(ckpt['classifier'])
    encoder.eval()
    classifier.eval()
    out = []
    for i in range(0, len(X), 512):
        x = X[i:i + 512].to(device).float()
        out.append(mapping[classifier(encoder(x)).argmax(1)])
    return torch.cat(out).cpu()


def main():
    args = parse_args()
    with open(os.path.join(args.run_dir, 'config.json')) as f:
        cfg = json.load(f)
    if args.data_dir:
        os.environ['SEED_DIR'] = args.data_dir
    elif cfg.get('data_dir'):
        os.environ['SEED_DIR'] = cfg['data_dir']
    os.environ['SEED_SESSION'] = str(cfg['session'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mapping = protocol_mine.trial_to_emotion().to(device)
    mapping_cpu = mapping.cpu()
    loss_key = 'cal_loss_final' if args.stage == 'final' else 'cal_loss'
    acc_key = 'acc_final' if args.stage == 'final' else 'acc_step3'

    rows = read_results(args.run_dir)
    targets = sorted({int(r['target']) for r in rows})
    out_rows = []

    for target in targets:
        pairs = [r for r in rows if int(r['target']) == target]
        pairs.sort(key=lambda r: float(r[loss_key]))
        chosen = pairs[:args.top_k]

        data_t, trial_t = Loading_Data.get_data_label(str(target))
        data_t = data_t.astype(np.float32)
        _, _, idx_tst = protocol_mine.split_by_protocol(trial_t, cfg['n_target_samples'], cfg['test_frac'],
                                                    cfg['gap'], cfg.get('protocol', 'fair'))
        X_test = torch.from_numpy(data_t[idx_tst])
        y_emotion = mapping_cpu[torch.from_numpy(trial_t[idx_tst]).long()]

        preds = []
        for r in chosen:
            path = os.path.join(args.run_dir, 'target_%d' % target,
                                'source_%d_%s.pt' % (int(r['source']), args.stage))
            preds.append(predict_emotion(path, X_test, device, mapping))
        stack = torch.stack(preds)
        vote = torch.mode(stack, dim=0).values
        acc_vote = (vote == y_emotion).float().mean().item()
        acc_top1 = float(chosen[0][acc_key])
        acc_mean_all = float(np.mean([float(r[acc_key]) for r in pairs]))

        out_rows.append({
            'target': target,
            'n_sources': len(pairs),
            'top_k': len(chosen),
            'chosen_sources': ' '.join(r['source'] for r in chosen),
            'acc_vote': round(acc_vote, 4),
            'acc_top1_by_cal': round(acc_top1, 4),
            'acc_mean_all_sources': round(acc_mean_all, 4),
            'n_test': len(idx_tst),
        })
        print("target %d vote %.4f top1 %.4f mean_all %.4f sources %s" % (
            target, acc_vote, acc_top1, acc_mean_all, out_rows[-1]['chosen_sources']))

    mean_vote = float(np.mean([r['acc_vote'] for r in out_rows]))
    mean_top1 = float(np.mean([r['acc_top1_by_cal'] for r in out_rows]))
    mean_all = float(np.mean([r['acc_mean_all_sources'] for r in out_rows]))
    print("mean vote %.4f  top1 %.4f  mean_all %.4f" % (mean_vote, mean_top1, mean_all))

    path = os.path.join(args.run_dir, 'voting_%s.csv' % args.stage)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("saved", path)


if __name__ == '__main__':
    main()
