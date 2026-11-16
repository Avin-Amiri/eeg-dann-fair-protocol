
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from models import main_models_seed_6
from models.main_models_seed_6 import Classifier, Encoder


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sources', type=str, default='1-15')
    p.add_argument('--targets', type=str, default='1-15')
    p.add_argument('--session', type=int, default=1)
    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='seed_6_fair')
    p.add_argument('--n_epoches_1', type=int, default=10)
    p.add_argument('--n_epoches_2', type=int, default=120)
    p.add_argument('--n_epoches_3', type=int, default=100)
    p.add_argument('--n_target_samples', type=int, default=12)   
    p.add_argument('--test_frac', type=float, default=0.3)
    p.add_argument('--gap', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=30)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def parse_ids(text):
    ids = []
    for part in text.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            ids.extend(range(int(a), int(b) + 1))
        elif part:
            ids.append(int(part))
    return ids


@torch.no_grad()
def predict(encoder, classifier, X, device, batch=512):
    out = []
    for i in range(0, len(X), batch):
        out.append(classifier(encoder(X[i:i + batch].to(device))).argmax(1).cpu())
    return torch.cat(out)


def emotion_acc(pred, y, mapping):
    return (mapping[pred] == mapping[y]).float().mean().item()


def make_batch(groups, flat_idx, k, device):
    """flat_idx: list of ints in [0, len(groups)*k). Returns x1, x2, group id."""
    gt = [i // k for i in flat_idx]
    pos = [i % k for i in flat_idx]
    x1 = torch.stack([groups[g][p][0].float() for g, p in zip(gt, pos)]).to(device)
    x2 = torch.stack([groups[g][p][1].float() for g, p in zip(gt, pos)]).to(device)
    return x1, x2, gt, pos


def run_pair(opt, num_s, num_t, Load_Data, protocol_mine, device):
    torch.manual_seed(opt.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(opt.seed)
    t0 = time.time()
    mapping = protocol_mine.trial_to_emotion().to(device)
    mapping_cpu = mapping.cpu()


    train_dataloader = Load_Data.source_loader(num_s)


    data_t, trial_t = Load_Data.get_data_label(num_t)
    idx_lab, _, idx_tst = protocol_mine.split_target(trial_t, opt.n_target_samples, opt.test_frac, opt.gap)
    X_t = torch.tensor(data_t[idx_lab], dtype=torch.float32)
    Y_t = torch.tensor(trial_t[idx_lab], dtype=torch.long)
    X_test = torch.tensor(data_t[idx_tst], dtype=torch.float32)
    Y_test = torch.tensor(trial_t[idx_tst], dtype=torch.long)

    classifier = Classifier().to(device)
    encoder = Encoder().to(device)
    discriminator = main_models_seed_6.DCD(input_features=128).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=0.001)

    acc_step1 = []
    for epoch in range(opt.n_epoches_1):
        for data, labels in train_dataloader:
            data = data.to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()
            loss = loss_fn(classifier(encoder(data)), labels)
            loss.backward()
            optimizer.step()
        acc_step1.append(emotion_acc(predict(encoder, classifier, X_test, device), Y_test, mapping_cpu))  

    X_s, Y_s = Load_Data.sample_data(num_s)

    sim = F.normalize(X_s, dim=1) @ F.normalize(X_t, dim=1).T
    sim = sim.masked_fill(Y_s[:, None] != Y_t[None, :], -2.0)
    nearest = sim.argmax(dim=1)
    X_s2 = 0.4 * X_s + 0.6 * X_t[nearest]
    Y_s2 = Y_s

    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    mini_batch = 30
    for epoch in range(opt.n_epoches_2):
        groups, _ = Load_Data.sample_groups(X_s, Y_s, X_s2, Y_s2, X_t, Y_t, seed=epoch)
        k = len(groups[1])
        order = torch.randperm(18 * k).tolist()
        losses = []
        for s in range(0, len(order) - mini_batch + 1, mini_batch):
            x1, x2, gt, _ = make_batch(groups, order[s:s + mini_batch], k, device)
            gt = torch.tensor(gt, dtype=torch.long, device=device)
            optimizer_D.zero_grad()
            with torch.no_grad():
                X_cat = torch.cat([encoder(x1), encoder(x2)], 1)
            loss = loss_fn(discriminator(X_cat), gt)
            loss.backward()
            optimizer_D.step()
            losses.append(loss.item())

    optimizer_g_h = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=0.0001)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    acc_step3 = []
    for epoch in range(opt.n_epoches_3):
        groups, groups_y = Load_Data.sample_groups(X_s, Y_s, X_s2, Y_s2, X_t, Y_t, seed=opt.n_epoches_2 + epoch)
        # groups order per version: G1..G6 ; target-containing ones are index 1,3,4,5 (G2,G4,G5,G6)
        sel = [1, 3, 4, 5, 7, 9, 10, 11, 13, 15, 16, 17]
        groups_2 = [groups[i] for i in sel]
        groups_y_2 = [groups_y[i] for i in sel]
        k = len(groups[1])

        order = torch.randperm(12 * k).tolist()
        for s in range(0, len(order) - mini_batch + 1, mini_batch):
            chunk = order[s:s + mini_batch]
            x1, x2, gt, pos = make_batch(groups_2, chunk, k, device)
            y1 = torch.stack([torch.as_tensor(groups_y_2[g][p][0]) for g, p in zip(gt, pos)]).long().to(device)
            y2 = torch.stack([torch.as_tensor(groups_y_2[g][p][1]) for g, p in zip(gt, pos)]).long().to(device)
            # same rule as before: even positions in the list -> 0, odd -> 2
            dcd_labels = torch.tensor([0 if g % 2 == 0 else 2 for g in gt], dtype=torch.long, device=device)
            optimizer_g_h.zero_grad()
            e1, e2 = encoder(x1), encoder(x2)
            y_pred_dcd = discriminator(torch.cat([e1, e2], 1))
            loss = loss_fn(classifier(e1), y1) + loss_fn(classifier(e2), y2) + 1 * loss_fn(y_pred_dcd, dcd_labels)
            loss.backward()
            optimizer_g_h.step()

        order = torch.randperm(18 * k).tolist()
        for s in range(0, len(order) - mini_batch + 1, mini_batch):
            x1, x2, gt, _ = make_batch(groups, order[s:s + mini_batch], k, device)
            gt = torch.tensor(gt, dtype=torch.long, device=device)
            optimizer_d.zero_grad()
            with torch.no_grad():
                X_cat = torch.cat([encoder(x1), encoder(x2)], 1)
            loss = loss_fn(discriminator(X_cat), gt)
            loss.backward()
            optimizer_d.step()

        pred = predict(encoder, classifier, X_test, device)
        acc_step3.append(emotion_acc(pred, Y_test, mapping_cpu))
        print("t%s s%s step3 epoch %d/%d test acc %.3f" % (num_t, num_s, epoch + 1, opt.n_epoches_3, acc_step3[-1]))

    pred_final = predict(encoder, classifier, X_test, device).numpy()
    return dict(acc_step1=np.array(acc_step1), acc_step3=np.array(acc_step3),
                acc_final=acc_step3[-1], acc_oracle=max(acc_step3),
                pred_final=pred_final, idx_test=idx_tst, y_test=Y_test.numpy(),
                n_labeled=len(idx_lab), n_test=len(idx_tst), time_sec=time.time() - t0)


def main():
    opt = get_args()
    os.environ['SEED_SESSION'] = str(opt.session)      
    if opt.data_dir:
        os.environ['SEED_DIR'] = opt.data_dir
    import Loading_Data as Load_Data
    import protocol_mine

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Load_Data.batch_size = opt.batch_size
    os.makedirs(opt.output_dir, exist_ok=True)
    csv_path = os.path.join(opt.output_dir, 'results.csv')
    fields = ['target', 'source', 'acc_final', 'acc_oracle', 'acc_step1_last', 'n_labeled', 'n_test', 'time_sec']
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(fields)

    for t in parse_ids(opt.targets):
        for s in parse_ids(opt.sources):
            if s == t:                                  
                continue
            npz = os.path.join(opt.output_dir, 't%d_s%d.npz' % (t, s))
            if os.path.exists(npz) and not opt.overwrite:
                continue
            r = run_pair(opt, str(s), str(t), Load_Data, protocol_mine, device)
            np.savez(npz, **r)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([t, s, '%.4f' % r['acc_final'], '%.4f' % r['acc_oracle'],
                                        '%.4f' % r['acc_step1'][-1], r['n_labeled'], r['n_test'],
                                        '%.1f' % r['time_sec']])
            print("DONE target %d source %d final %.4f oracle %.4f" % (t, s, r['acc_final'], r['acc_oracle']))


if __name__ == '__main__':
    main()
