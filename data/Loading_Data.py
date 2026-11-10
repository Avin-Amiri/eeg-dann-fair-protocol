import glob
import os
import numpy as np
import scipy.io as iso
import torch

batch_size = 30
_cache = {}


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


def _normalize(x):
    for k in range(x.shape[2]):
        m = np.mean(x[:, :, k])
        x[:, :, k] = 2 * (x[:, :, k] - m) / m
    return x


def _stack(data_file, first, last):
    parts = [np.squeeze(data_file['de_LDS' + str(i)]) for i in range(first, last + 1)]
    return np.concatenate(parts, axis=1).transpose(1, 0, 2)


def get_data_label(input_file="2"):
    dataset_dir = os.environ.get("SEED_DIR", ".")
    session = int(os.environ.get("SEED_SESSION", "1"))
    key = (dataset_dir, session, str(input_file))
    if key in _cache:
        data, labels = _cache[key]
        return data.copy(), labels.copy()
    pattern = os.path.join(dataset_dir, str(input_file) + "_*.mat")
    subject_files = sorted(glob.glob(pattern), key=lambda f: os.path.basename(f).split("_")[1])
    if len(subject_files) < session:
        raise FileNotFoundError("SEED file not found: " + pattern + " session " + str(session))
    data_file = iso.loadmat(subject_files[session - 1])
    train_data = _normalize(_stack(data_file, 1, 9))
    test_data = _normalize(_stack(data_file, 10, 15))
    data = np.vstack((train_data, test_data))
    data = np.reshape(data, (data.shape[0], -1))
    labels = []
    for i in range(15):
        n = np.squeeze(data_file['de_LDS' + str(i + 1)]).shape[1]
        labels.extend([i] * n)
    labels = np.array(labels)
    _cache[key] = (data, labels)
    return data.copy(), labels.copy()


def source_loader(source_dataset_name):
    source_data, source_label = get_data_label(source_dataset_name)
    return torch.utils.data.DataLoader(
        dataset=MyDataset(source_data, source_label),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=0)


def target_loader(target_dataset_name):
    target_data, target_label = get_data_label(target_dataset_name)
    return torch.utils.data.DataLoader(
        dataset=MyDataset(target_data, target_label),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=0)


def sample_data(source_dataset_name):
    data, label = get_data_label(source_dataset_name)
    inds = torch.randperm(len(data)).numpy()
    X = torch.tensor(data[inds], dtype=torch.float32)
    Y = torch.tensor(label[inds], dtype=torch.long)
    return X, Y


_SHIFT = [(10, 15), (20, 25), (50, 70), (95, 115), (140, 160), (185, 205), (230, 250), (270, 285), (300, 310)]


def _shift_source(X_s):
    X_s1 = X_s.clone()
    for a, b in _SHIFT:
        X_s1[:-1, a:b] = X_s[1:, a:b]
    return X_s1


def create_groups(X_s, Y_s, X_s2, Y_s2, X_t, Y_t, seed=1):
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)
    n = X_t.shape[0]
    classes = torch.unique(Y_t)
    classes = classes[torch.randperm(len(classes))]
    class_num = classes.shape[0]
    shot = n // class_num
    X_s1 = _shift_source(X_s)

    def s_idxs(c):
        idx = torch.nonzero(Y_s.eq(int(c)))
        return idx[torch.randperm(len(idx))][:shot * 2].squeeze()

    def t_idxs(c):
        return torch.nonzero(Y_t.eq(int(c)))[:shot].squeeze()

    sm = torch.stack(list(map(s_idxs, classes)))
    tm = torch.stack(list(map(t_idxs, classes)))

    def build(Xa):
        G = [[] for _ in range(6)]
        Yg = [[] for _ in range(6)]
        for i in range(class_num):
            for j in range(shot):
                G[0].append((Xa[sm[i][j * 2]], Xa[sm[i][j * 2 + 1]]))
                Yg[0].append((Y_s[sm[i][j * 2]], Y_s[sm[i][j * 2 + 1]]))
                G[1].append((Xa[sm[i][j]], X_t[tm[i][j]]))
                Yg[1].append((Y_s[sm[i][j]], Y_t[tm[i][j]]))
                G[2].append((Xa[sm[i % 2][j]], Xa[sm[(i + 1) % 2][j]]))
                Yg[2].append((Y_s[sm[i % 2][j]], Y_s[sm[(i + 1) % 2][j]]))
                G[3].append((Xa[sm[i % 2][j]], X_t[tm[(i + 1) % 2][j]]))
                Yg[3].append((Y_s[sm[i % 2][j]], Y_t[tm[(i + 1) % 2][j]]))
        for i in range(class_num):
            for j in range(shot):
                G[4].append((X_t[tm[i][j]], X_t[tm[i][(j + 1) % shot]]))
                Yg[4].append((Y_t[tm[i][j]], Y_t[tm[i][(j + 1) % shot]]))
                k = j if i == 0 else (j + 1) % shot
                G[5].append((X_t[tm[i][j]], X_t[tm[(i + 1) % 2][k]]))
                Yg[5].append((Y_t[tm[i][j]], Y_t[tm[(i + 1) % 2][k]]))
        return G, Yg

    groups, groups_y = [], []
    for Xa in (X_s, X_s1, X_s2):
        G, Yg = build(Xa)
        groups.extend(G)
        groups_y.extend(Yg)
    for g in groups:
        assert len(g) == n
    return groups, groups_y


def sample_groups(X_s, Y_s, X_s2, Y_s2, X_t, Y_t, seed=1):
    return create_groups(X_s, Y_s, X_s2, Y_s2, X_t, Y_t, seed=seed)
