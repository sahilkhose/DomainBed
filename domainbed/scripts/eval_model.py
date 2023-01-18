# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

## For latent visualization
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def _get_domainbed_dataloaders(args, dataset, hparams):
    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]
    
    loaders = (train_loaders, uda_loaders, eval_loaders)
    others = (eval_weights, eval_loader_names, in_splits)
    return loaders, others


def _get_ood_validation_dataloaders(args, dataset, hparams):
    # A separate set of validation environments (specified by "--val_envs") is used for
    # model selection. The data from training environments are all used for training,
    # and the data from test environments are all used for testing. The holdout data
    # of which the size is specified by "--holdout_fraction" are now a sample of
    # the training data and are used to compute training accuracy, equivalent to the
    # training-environemnt in-split accuarcy in other model selection methods.
    if args.task == 'domain_adaptation' or args.uda_holdout_fraction > 0:
        raise NotImplementedError

    in_splits = []
    out_splits = []

    for env_i, env in enumerate(dataset):
        if hparams['class_balanced']:
            weights = misc.make_weights_for_balanced_classes(env)
        else:
            weights = None
        if env_i in args.val_envs:
            in_splits.append((None, None))  # dummy placeholder
            out_splits.append((env, weights))
        elif env_i in args.test_envs:
            in_splits.append((None, None))  # dummy placeholder
            out_splits.append((env, weights))
        else:
            out, _ = misc.split_dataset(env,
                int(len(env)*args.holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))
            in_splits.append((env, weights))
            # add a small sample to check training accuracy
            if hparams['class_balanced']:
                out_weights = misc.make_weights_for_balanced_classes(out)
            else:
                out_weights = None
            out_splits.append((out, out_weights))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs + args.val_envs]

    uda_loaders = []

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in out_splits]   # in splits are removed to save computation time
    eval_weights = [None for _, weights in out_splits]

    eval_loader_names = ['env{}_out'.format(i) for i in range(len(dataset))]

    loaders = (train_loaders, uda_loaders, eval_loaders)
    others = (eval_weights, eval_loader_names, in_splits)
    return loaders, others


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--val_envs', type=int, nargs='+', default=[],
        help='Environments for OOD-validation model selection.')
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--load', type=str, default="train_output/model.pkl")
    args = parser.parse_args()
    # writer = SummaryWriter(comment=f"_{args.algorithm}") # Tensorboard visualization
    
    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None
    algorithm_dict = torch.load(args.load)['model_dict']
    print(f"{args.algorithm} loading from {args.load} evaluating on {args.dataset} {args.test_envs}")

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        no_aug_envs = args.test_envs + args.val_envs  # no data augmentations
        dataset = vars(datasets)[args.dataset](args.data_dir, no_aug_envs, hparams)
    else:
        raise NotImplementedError

    if args.val_envs:
        loaders, others = _get_ood_validation_dataloaders(args, dataset, hparams)
    else:
        loaders, others = _get_domainbed_dataloaders(args, dataset, hparams)
    train_loaders, uda_loaders, eval_loaders = loaders
    eval_weights, eval_loader_names, in_splits = others

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs) - len(args.val_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])
    steps_per_epoch = min([len(env)/hparams['batch_size'] for env, _ in in_splits
                          if env is not None])
    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    last_results_keys = None
    for step in range(start_step, n_steps):
        print("__"*40)
        print("Evaluation started...")
        if step == 0:
            ## Print Accuracy
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)
            
            def plot_tsne(embedding, label, num_categories, save_name):
                ## Got the embeddings, dimensionality reduction now
                tsne = TSNE(2, verbose=1)
                tsne_proj = tsne.fit_transform(embedding, label)
                # Plot those points as a scatter plot and label them based on the pred labels
                cmap = cm.get_cmap('tab20')
                fig, ax = plt.subplots(figsize=(8,8))
                for lab in range(num_categories):
                    indices = label==lab
                    ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(cmap(lab)).reshape(1, 4), label=lab, alpha=0.5)
                ax.legend(fontsize='large', markerscale=2)
                folder_name = args.load.split(".")[-2].split("/")[-1]
                os.makedirs(os.path.join("tsne_results", folder_name), exist_ok=True)
                file_name = os.path.join("tsne_results", folder_name, save_name) 
                plt.savefig(file_name) # ./tsne_results/[model name from .pkl]/env.png
                print(f"Saving... {file_name}")
                print("__"*40)
            
            ## Data visualization -- CLASSES
            print("__"*40)
            print("CLASSES TSNE")
            print("__"*40)
            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals: 
                print(name)
                algorithm.eval()
                with torch.no_grad():
                    embeddings = []
                    all_y_pred = []
                    for x, y in tqdm(loader):
                        x = x.to(device)
                        y = y.to(device)
                        p = algorithm.predict(x)
                        emb = algorithm.featurizer(x)
                        # all_y_pred.extend(p.argmax(1).detach().cpu().numpy()) # prediction
                        all_y_pred.extend(y.detach().cpu().numpy()) # label
                        embeddings.extend(emb.detach().cpu().numpy())
                    all_y_pred = np.array(all_y_pred)
                    embeddings = np.array(embeddings)
                    plot_tsne(embeddings, all_y_pred, num_categories=7, save_name=f"{name}.png")
            
            ## Data visualization -- _in DOMAINS and _in CLASSES
            print("__"*40)
            print("DOMAINS and CLASSES TSNE")
            print("__"*40)
            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            main_embeddings = []
            main_domains = []
            main_all_y_pred = []
            for name, loader, weights in evals: 
                print(name)
                algorithm.eval()
                if "in" in name: # just run on "_in" envs
                    with torch.no_grad():
                        embeddings = []
                        all_y_pred = []
                        for x, y in tqdm(loader):
                            x = x.to(device)
                            y = y.to(device)
                            p = algorithm.predict(x)
                            emb = algorithm.featurizer(x)
                            embeddings.extend(emb.detach().cpu().numpy())
                            all_y_pred.extend(y.detach().cpu().numpy())
                        embeddings = np.array(embeddings)
                        domain_labels = np.ones(embeddings.shape[0]) * int(name.split("env")[1].split("_")[0]) 
                        main_embeddings.append(embeddings)
                        main_domains.append(domain_labels)
                        main_all_y_pred.append(all_y_pred)
                        
            main_embeddings = np.concatenate(main_embeddings, axis=0)
            main_domains = np.concatenate(main_domains, axis=0)
            main_all_y_pred = np.concatenate(main_all_y_pred, axis=0)
            plot_tsne(main_embeddings, main_domains, num_categories=4, save_name=f"all_domains.png")
            plot_tsne(main_embeddings, main_all_y_pred, num_categories=7, save_name=f"all_classes.png")
            
            exit()

    # with open(os.path.join(args.output_dir, 'done'), 'w') as f:
    #     f.write('done')
