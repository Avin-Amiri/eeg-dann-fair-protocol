import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import Loading_Data
from protocols import protocol_mine
from models import main_models_seed_6
from models.main_models_seed_6 import Classifier, Encoder

PAIR_GROUPS = (1, 3, 4, 5, 7, 9, 10, 11, 13, 15, 16, 17)
CSV_FIELDS = ['target', 'source', 'acc_step3', 'acc_final', 'cal_loss', 'cal_loss_final',
              'acc_oracle', 'pl_first', 'pl_last', 'n_labeled', 'n_unlabeled', 'n_test', 'time_sec']


def parse_ids(text):
    ids = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    return ids


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sources', type=str, default='1-15')
    p.add_argument('--targets', type=str, default='1-15')
    p.add_argument('--session', type=int, default=1)
    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='results_seed6')
    p.add_argument('--n_target_samples', type=int, default=12)
    p.add_argument('--test_frac', type=float, default=0.3)
    p.add_argument('--gap', type=int, default=10)
    p.add_argument('--n_epoches_1', type=int, default=10)
    p.add_argument('--n_epoches_2', type=int, default=120)
    p.add_argument('--n_epoches_3', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=30)
    p.add_argument('--pl_epochs', type=int, default=40)
    p.add_argument('--pl_threshold', type=float, default=0.9)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--protocol', type=str, default='fair', choices=['fair', 'base'])
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def build_source_mix(X_s, Y_s, X_t, Y_t, w=0.4):
    sim = F.normalize(X_s, dim=1) @ F.normalize(X_t, dim=1).T
    sim = sim.masked_fill(Y_s[:, None] != Y_t[None, :], -2.0)
    nearest = sim.argmax(dim=1)
    return w * X_s + (1 - w) * X_t[nearest]


def dcd_batch(groups, idx, device):
    k = len(groups[0])
    gt = [i // k for i in idx]
    pos = [i % k for i in idx]
    x1 = torch.stack([groups[g][p][0].float() for g, p in zip(gt, pos)]).to(device)
    x2 = torch.stack([groups[g][p][1].float() for g, p in zip(gt, pos)]).to(device)
    y = torch.tensor(gt, dtype=torch.long, device=device)
    return x1, x2, y


def train_dcd(encoder, discriminator, opt, groups, batch, device, loss_fn):
    k = len(groups[0])
    order = torch.randperm(len(groups) * k).tolist()
    losses = []
    for s in range(0, len(order) - batch + 1, batch):
        x1, x2, y = dcd_batch(groups, order[s:s + batch], device)
        with torch.no_grad():
            z = torch.cat([encoder(x1), encoder(x2)], 1)
        opt.zero_grad()
        loss = loss_fn(discriminator(z), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def train_encoder_step(encoder, classifier, discriminator, opt, groups, groups_y, batch, device, loss_fn):
    g2 = [groups[i] for i in PAIR_GROUPS]
    gy = [groups_y[i] for i in PAIR_GROUPS]
    k = len(g2[0])
    order = torch.randperm(len(g2) * k).tolist()
    for s in range(0, len(order) - batch + 1, batch):
        idx = order[s:s + batch]
        gt = [i // k for i in idx]
        pos = [i % k for i in idx]
        x1 = torch.stack([g2[g][p][0].float() for g, p in zip(gt, pos)]).to(device)
        x2 = torch.stack([g2[g][p][1].float() for g, p in zip(gt, pos)]).to(device)
        ya = torch.tensor([int(gy[g][p][0]) for g, p in zip(gt, pos)], dtype=torch.long, device=device)
        yb = torch.tensor([int(gy[g][p][1]) for g, p in zip(gt, pos)], dtype=torch.long, device=device)
        dcd = torch.tensor([0 if g % 2 == 0 else 2 for g in gt], dtype=torch.long, device=device)
        opt.zero_grad()
        e1 = encoder(x1)
        e2 = encoder(x2)
        loss = (loss_fn(classifier(e1), ya) + loss_fn(classifier(e2), yb)
                + loss_fn(discriminator(torch.cat([e1, e2], 1)), dcd))
        loss.backward()
        opt.step()


@torch.no_grad()
def calibration_loss(encoder, classifier, X, Y, device, loss_fn):
    encoder.eval()
    classifier.eval()
    loss = loss_fn(classifier(encoder(X.to(device))), Y.to(device)).item()
    encoder.train()
    classifier.train()
    return loss


def run_pair(args, source, target, device, pair_dir):
    start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = Loading_Data.source_loader(str(source))
    data_t, trial_t = Loading_Data.get_data_label(str(target))
    data_t = data_t.astype(np.float32)
    idx_lab, idx_unl, idx_tst = protocol_mine.split_by_protocol(trial_t, args.n_target_samples, args.test_frac,
                                                                 args.gap, args.protocol)
    X_t = torch.from_numpy(data_t[idx_lab])
    Y_t = torch.from_numpy(trial_t[idx_lab]).long()
    X_unl = torch.from_numpy(data_t[idx_unl])
    X_test = torch.from_numpy(data_t[idx_tst])
    Y_test = torch.from_numpy(trial_t[idx_tst]).long()

    encoder = Encoder().to(device)
    classifier = Classifier().to(device)
    discriminator = main_models_seed_6.DCD(input_features=128).to(device)

    opt_src = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=0.001)
    for epoch in range(args.n_epoches_1):
        for data, labels in train_loader:
            data = data.float().to(device)
            labels = labels.long().to(device)
            opt_src.zero_grad()
            loss = loss_fn(classifier(encoder(data)), labels)
            loss.backward()
            opt_src.step()
        cal_acc = protocol_mine.emotion_accuracy(encoder, classifier, X_t, Y_t, device)
        print("step1 epoch %d/%d calibration acc %.3f" % (epoch + 1, args.n_epoches_1, cal_acc))

    X_s, Y_s = Loading_Data.sample_data(str(source))
    X_s2 = build_source_mix(X_s, Y_s, X_t, Y_t)

    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    for epoch in range(args.n_epoches_2):
        groups, _ = Loading_Data.sample_groups(X_s, Y_s, X_s2, Y_s, X_t, Y_t, seed=epoch + (args.seed - 1) * 1000)
        loss_d = train_dcd(encoder, discriminator, opt_d, groups, args.batch_size, device, loss_fn)
        print("step2 epoch %d/%d discriminator loss %.3f" % (epoch + 1, args.n_epoches_2, loss_d))

    opt_gh = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=0.0001)
    opt_d3 = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    acc_oracle = -1.0  # only for protocol=base: best test accuracy over epochs (what the base code reports)
    for epoch in range(args.n_epoches_3):
        groups, groups_y = Loading_Data.sample_groups(X_s, Y_s, X_s2, Y_s, X_t, Y_t,
                                                           seed=args.n_epoches_2 + epoch + (args.seed - 1) * 1000)
        train_encoder_step(encoder, classifier, discriminator, opt_gh, groups, groups_y,
                           args.batch_size, device, loss_fn)
        train_dcd(encoder, discriminator, opt_d3, groups, args.batch_size, device, loss_fn)
        cal_acc = protocol_mine.emotion_accuracy(encoder, classifier, X_t, Y_t, device)
        print("step3 epoch %d/%d calibration acc %.3f" % (epoch + 1, args.n_epoches_3, cal_acc))
        if args.protocol == 'base':
            acc_oracle = max(acc_oracle, protocol_mine.emotion_accuracy(encoder, classifier, X_test, Y_test, device))

    acc_step3 = protocol_mine.emotion_accuracy(encoder, classifier, X_test, Y_test, device)
    cal_loss = calibration_loss(encoder, classifier, X_t, Y_t, device, loss_fn)
    torch.save({'encoder': encoder.state_dict(), 'classifier': classifier.state_dict()},
               os.path.join(pair_dir, 'source_%d_step3.pt' % source))

    used = protocol_mine.pseudo_label_adaptation(encoder, classifier, X_t, Y_t, X_unl, X_s, Y_s, device,
                                            epochs=args.pl_epochs, threshold=args.pl_threshold,
                                            seed=args.seed)
    acc_final = protocol_mine.emotion_accuracy(encoder, classifier, X_test, Y_test, device)
    cal_loss_final = calibration_loss(encoder, classifier, X_t, Y_t, device, loss_fn)
    torch.save({'encoder': encoder.state_dict(), 'classifier': classifier.state_dict()},
               os.path.join(pair_dir, 'source_%d_final.pt' % source))

    row = {
        'target': target,
        'source': source,
        'acc_step3': round(acc_step3, 4),
        'acc_final': round(acc_final, 4),
        'cal_loss': round(cal_loss, 4),
        'cal_loss_final': round(cal_loss_final, 4),
        'acc_oracle': round(acc_oracle, 4),
        'pl_first': round(float(used[0]), 4) if used else 0.0,
        'pl_last': round(float(used[-1]), 4) if used else 0.0,
        'n_labeled': len(idx_lab),
        'n_unlabeled': len(idx_unl),
        'n_test': len(idx_tst),
        'time_sec': round(time.time() - start, 1),
    }
    np.savez(os.path.join(pair_dir, 'source_%d.npz' % source), **row)
    print("target %d source %d  step3: %.3f  final: %.3f" % (target, source, acc_step3, acc_final))
    return row


def write_results(run_root):
    rows = []
    for path in sorted(glob.glob(os.path.join(run_root, 'target_*', 'source_*.npz'))):
        d = np.load(path)
        rows.append({k: (d[k].item() if k in d.files else '') for k in CSV_FIELDS})
    with open(os.path.join(run_root, 'results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    if args.data_dir:
        os.environ['SEED_DIR'] = args.data_dir
    os.environ['SEED_SESSION'] = str(args.session)
    Loading_Data.batch_size = args.batch_size
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

        # session 1 keeps the old folder name (n12_seed1); other sessions get a suffix
    run_name = 'n%d_seed%d' % (args.n_target_samples, args.seed)
    if args.session != 1:
        run_name += '_s%d' % args.session
    if args.protocol != 'fair':
        run_name += '_' + args.protocol
    run_root = os.path.join(args.output_dir, run_name)
    os.makedirs(run_root, exist_ok=True)
    cfg_path = os.path.join(run_root, 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            old = json.load(f)
        for k in ('session', 'seed', 'n_target_samples', 'protocol'):
            if k in old and old[k] != getattr(args, k):
                raise SystemExit('config mismatch in %s: %s=%s, now %s' % (run_root, k, old[k], getattr(args, k)))
    print('output folder:', run_root)
    with open(cfg_path, 'w') as f:
        json.dump(vars(args), f, indent=2)


    for target in parse_ids(args.targets):
        for source in parse_ids(args.sources):
            if source == target:
                continue
            pair_dir = os.path.join(run_root, 'target_%d' % target)
            out_file = os.path.join(pair_dir, 'source_%d.npz' % source)
            if os.path.exists(out_file) and not args.overwrite:
                print("skip target %d source %d (already done)" % (target, source))
                continue
            os.makedirs(pair_dir, exist_ok=True)
            run_pair(args, source, target, device, pair_dir)
            write_results(run_root)


if __name__ == '__main__':
    main()
