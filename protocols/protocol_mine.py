import numpy as np
import torch
import torch.nn.functional as F

POSITIVE_TRIALS = (0, 5, 8, 9, 13)
NEUTRAL_TRIALS = (1, 4, 7, 10, 12)


def trial_to_emotion():
    m = np.zeros(15, dtype=np.int64)
    m[list(POSITIVE_TRIALS)] = 2
    m[list(NEUTRAL_TRIALS)] = 1
    return torch.from_numpy(m)


def split_target(trial_ids, n_labeled, test_frac=0.3, gap=10):
    trial_ids = np.asarray(trial_ids)
    lab, unl, tst = [], [], []
    for t in np.unique(trial_ids):
        idx = np.flatnonzero(trial_ids == t)
        n = len(idx)
        n_test = int(round(n * test_frac))
        test_start = n - n_test
        if n_labeled > test_start - gap:
            raise ValueError(
                "n_labeled=%d is too large for trial %d with %d samples" % (n_labeled, int(t), n)
            )
        pool_end = max(n_labeled, test_start - gap)
        lab.append(idx[:n_labeled])
        unl.append(idx[n_labeled:pool_end])
        tst.append(idx[test_start:])
    lab, unl, tst = (np.concatenate(a) for a in (lab, unl, tst))
    assert not (set(lab) & set(unl))
    assert not (set(lab) & set(tst))
    assert not (set(unl) & set(tst))
    return lab, unl, tst


def split_base(trial_ids, n_labeled):
    # base-code protocol: first n windows of each trial are labelled,
    # the test set is the WHOLE target subject (labelled windows included), no unlabelled pool
    trial_ids = np.asarray(trial_ids)
    lab = np.concatenate([np.flatnonzero(trial_ids == t)[:n_labeled] for t in np.unique(trial_ids)])
    return lab, np.array([], dtype=np.int64), np.arange(len(trial_ids))


def split_by_protocol(trial_ids, n_labeled, test_frac=0.3, gap=10, protocol='fair'):
    if protocol == 'base':
        return split_base(trial_ids, n_labeled)
    return split_target(trial_ids, n_labeled, test_frac, gap)


@torch.no_grad()
def emotion_accuracy(encoder, classifier, X, Y, device, batch=512):
    encoder.eval()
    classifier.eval()
    mapping = trial_to_emotion().to(device)
    hit = 0
    for i in range(0, len(X), batch):
        x = X[i:i + batch].to(device).float()
        y = Y[i:i + batch].to(device).long()
        pred = classifier(encoder(x)).argmax(1)
        hit += (mapping[pred] == mapping[y]).sum().item()
    encoder.train()
    classifier.train()
    return hit / max(len(X), 1)


def _emotion_probs(p15, mapping_onehot):
    return p15 @ mapping_onehot


def _perturb(x, noise_std, mask_p):
    x = x + noise_std * torch.randn_like(x)
    keep = (torch.rand_like(x) > mask_p).float()
    return x * keep


def pseudo_label_adaptation(encoder, classifier, X_lab, Y_lab, X_unl, X_src, Y_src, device,
                            epochs=40, batch=64, lr=5e-5, threshold=0.9,
                            noise_std=0.1, mask_p=0.1, w_unl=1.0, w_src=0.5, seed=0):
    stats = []
    if epochs <= 0 or len(X_unl) == 0:
        return stats
    gen = torch.Generator().manual_seed(seed)
    onehot = F.one_hot(trial_to_emotion(), 3).float().to(device)
    params = list(encoder.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    X_lab, Y_lab = X_lab.float(), Y_lab.long()
    X_unl = X_unl.float()
    X_src, Y_src = X_src.float(), Y_src.long()
    batch = min(batch, len(X_unl))
    eps = 1e-8
    encoder.train()
    classifier.train()

    for ep in range(epochs):
        order = torch.randperm(len(X_unl), generator=gen)
        used, total = 0, 0
        for start in range(0, len(order) - batch + 1, batch):
            xu = X_unl[order[start:start + batch]].to(device)
            li = torch.randint(0, len(X_lab), (batch,), generator=gen)
            si = torch.randint(0, len(X_src), (batch,), generator=gen)
            xl, yl = X_lab[li].to(device), Y_lab[li].to(device)
            xs, ys = X_src[si].to(device), Y_src[si].to(device)

            with torch.no_grad():
                pu = _emotion_probs(classifier(encoder(xu)), onehot)
                conf, pseudo = pu.max(1)
                sel = conf >= threshold

            loss_l = F.nll_loss(torch.log(classifier(encoder(xl)) + eps), yl)
            loss_s = F.nll_loss(torch.log(classifier(encoder(xs)) + eps), ys)
            loss = loss_l + w_src * loss_s

            if sel.any():
                xa = _perturb(xu[sel], noise_std, mask_p)
                pa = _emotion_probs(classifier(encoder(xa)), onehot)
                loss_u = F.nll_loss(torch.log(pa + eps), pseudo[sel])
                loss = loss + w_unl * loss_u

            opt.zero_grad()
            loss.backward()
            opt.step()
            used += int(sel.sum())
            total += batch
        stats.append(used / max(total, 1))
    return stats
