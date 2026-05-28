import argparse
import pickle
import time
import numpy as np

from master import MASTERModel

# Please install qlib first before load the data.


def parse_args():
    p = argparse.ArgumentParser(description="MASTER: train or test")
    p.add_argument('mode', choices=['train', 'test'],
                   help="train: fit 5 seeds and save checkpoints; test: load checkpoint and evaluate")
    p.add_argument('--universe', default='csi300', choices=['csi300', 'csi800'])
    p.add_argument('--prefix', default='opensource', choices=['opensource', 'original'],
                   help="which training data to use (also part of checkpoint filename)")
    p.add_argument('--seeds', type=int, nargs='+', default=None,
                   help="seeds to run; default [0,1,2,3,4] for train, [0] for test")
    p.add_argument('--n-epoch', type=int, default=40, help="train only: max epochs per seed")
    p.add_argument('--gpu', type=int, default=0)
    return p.parse_args()


def build_model(universe, prefix, seed, n_epoch, gpu):
    d_feat = 158
    d_model = 256
    t_nhead = 4
    s_nhead = 2
    dropout = 0.5
    gate_input_start_index = 158
    gate_input_end_index = 221
    beta = 5 if universe == 'csi300' else 2
    lr = 1e-5
    train_stop_loss_thred = 0.95

    return MASTERModel(
        d_feat=d_feat, d_model=d_model, t_nhead=t_nhead, s_nhead=s_nhead,
        T_dropout_rate=dropout, S_dropout_rate=dropout,
        beta=beta,
        gate_input_start_index=gate_input_start_index,
        gate_input_end_index=gate_input_end_index,
        n_epochs=n_epoch, lr=lr, GPU=gpu, seed=seed,
        train_stop_loss_thred=train_stop_loss_thred,
        save_path='model', save_prefix=f'{universe}_{prefix}',
    )


def main():
    args = parse_args()
    universe, prefix, mode = args.universe, args.prefix, args.mode
    seeds = args.seeds or ([0, 1, 2, 3, 4] if mode == 'train' else [0])

    predict_data_dir = 'data/opensource'
    with open(f'{predict_data_dir}/{universe}_dl_test.pkl', 'rb') as f:
        dl_test = pickle.load(f)

    dl_train = dl_valid = None
    if mode == 'train':
        with open(f'data/{prefix}/{universe}_dl_train.pkl', 'rb') as f:
            dl_train = pickle.load(f)
        with open(f'{predict_data_dir}/{universe}_dl_valid.pkl', 'rb') as f:
            dl_valid = pickle.load(f)

    print("Data Loaded.")

    ic, icir, ric, ricir = [], [], [], []

    for seed in seeds:
        model = build_model(universe, prefix, seed, args.n_epoch, args.gpu)

        if mode == 'train':
            start = time.time()
            model.fit(dl_train, dl_valid)
            print("Model Trained.")
            _, metrics = model.predict(dl_test)
            print('Seed: {:d} time cost : {:.2f} sec'.format(seed, time.time() - start))
        else:
            param_path = f'model/{universe}_{prefix}_{seed}.pkl'
            print(f'Model Loaded from {param_path}')
            model.load_param(param_path)
            _, metrics = model.predict(dl_test)

        print(metrics)
        ic.append(metrics['IC'])
        icir.append(metrics['ICIR'])
        ric.append(metrics['RIC'])
        ricir.append(metrics['RICIR'])

    print("IC: {:.4f} pm {:.4f}".format(np.mean(ic), np.std(ic)))
    print("ICIR: {:.4f} pm {:.4f}".format(np.mean(icir), np.std(icir)))
    print("RIC: {:.4f} pm {:.4f}".format(np.mean(ric), np.std(ric)))
    print("RICIR: {:.4f} pm {:.4f}".format(np.mean(ricir), np.std(ricir)))


if __name__ == '__main__':
    main()
