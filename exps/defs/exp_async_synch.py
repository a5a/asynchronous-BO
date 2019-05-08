"""
Contains the async BO experiment definition *without* any info of the task
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import pylab as plt

from exps.exp_utils import apply_general_settings, \
    get_default_optimisation_params, create_async_bo_instance, \
    get_commit_hash, create_synch_bo_instance, get_async_synch_from_args
from ml_utils import timed_print as print
from ml_utils.misc import time_limited_df_to_pickle
from ml_utils.models import GP


def exp_async_synch(args=None, sampler=None, true_min_val=None,
                    x_init=None, y_init=None, x_lim=None, foldername=None,
                    hp_bounds=None, restart_bounds=None, hyper_priors=None,
                    n_iter=None, starting_jobs=None, async_interface=None,
                    force_run=None, debug=None):
    """
    Using keyword args for clearer calling of the func.
    All args that are necessary are asserted to not be None
    """
    assert args is not None
    assert sampler is not None
    assert true_min_val is not None
    assert x_init is not None
    assert y_init is not None
    assert x_lim is not None
    assert foldername is not None
    assert hp_bounds is not None
    assert restart_bounds is not None
    assert hyper_priors is not None
    assert n_iter is not None
    assert async_interface is not None
    assert force_run is not None
    assert debug is not None

    exp_type = get_async_synch_from_args(args)

    # *********   Convert args   ******** #
    seed, proc_name, batch_size, ard_flag, n_workers, kern_choice, \
    kern_func, acq_dict, opt_frequency = apply_general_settings(args)

    # *********   Optimisations   ******** #
    y_min_opt_params, acq_opt_params, min_acq_opt_params, gp_opt_params, \
    optimise_surrogate_model = get_default_optimisation_params(hp_bounds,
                                                               restart_bounds)

    kern_gp = kern_func(x_lim.shape[0], variance=np.var(y_init),
                        lengthscale=0.2, ARD=ard_flag)
    surrogate = GP(x_init, y_init, kern_gp, lik_variance=1e-6,
                   lik_variance_fixed=True, opt_params=gp_opt_params,
                   hyper_priors=hyper_priors)
    # surrogate.optimize()

    # *********   Paths   ******** #
    if exp_type == 'async':
        postfix = 'results/asyncbo'
    elif exp_type == 'synch':
        postfix = 'results/synchbo'
    else:
        raise NotImplementedError

    if args.machine == 0:  # local
        folder = os.path.join(f'../{postfix}/', foldername)
    elif args.machine == 1:  # arc
        raise NotImplementedError
    else:
        folder = os.path.join(f'../{postfix}', foldername)

    if exp_type == 'async':
        # synth time vs real exp
        if hasattr(args, 'timer'):
            fname = f'asyncst_{args.proc}_{acq_dict["type"]}_' \
                f'{kern_choice}_{sampler.__name__}_{n_workers}_' \
                f'{batch_size}_{seed}_{n_iter}_' \
                f't{args.timer}.pkl'
        else:
            fname = f'asyncbo_{args.proc}_{acq_dict["type"]}_' \
                f'{kern_choice}_{sampler.__name__}_{n_workers}_' \
                f'{batch_size}_{seed}_{n_iter}.pkl'
    elif exp_type == 'synch':
        # synth time vs real exp
        if hasattr(args, 'timer'):
            fname = f'synchst_{args.proc}_{acq_dict["type"]}_' \
                f'{kern_choice}_{sampler.__name__}_{n_workers}_' \
                f'{batch_size}_{seed}_{n_iter}_' \
                f't{args.timer}.pkl'
        else:
            fname = f'synchbo_{args.proc}_{acq_dict["type"]}_' \
                f'{kern_choice}_{sampler.__name__}_{n_workers}_' \
                f'{batch_size}_{seed}_{n_iter}.pkl'
    else:
        raise NotImplementedError

    save_fname = os.path.join(folder, fname)

    if os.path.exists(save_fname) and not (debug or force_run):
        print('*** WARNING:', save_fname, 'exists already on disk! '
                                          'Skipping...')
        sys.exit()

    # Avoids multiple instances of this code trying to create the folder at
    # the same time and then crashing
    time.sleep(np.random.uniform(0, 2))

    if not os.path.exists(folder):
        os.makedirs(folder)

    # *********   BO   ******** #

    df_all_exps = None

    # Run BayesOpt
    print(f"Starting exp {proc_name} with ({sampler.__name__}, {seed})")

    kwargs = dict()

    kwargs['acq_dict'] = acq_dict
    kwargs['y_min_opt_params'] = y_min_opt_params
    kwargs['acq_opt_params'] = acq_opt_params
    kwargs['min_acq_opt_params'] = min_acq_opt_params
    kwargs['offset_acq'] = True
    kwargs['n_bo_steps'] = n_iter
    kwargs['optimise_surrogate_model'] = optimise_surrogate_model
    kwargs['track_cond_k'] = False
    kwargs['verbose'] = 1
    kwargs['batch_size'] = batch_size
    kwargs['starting_jobs'] = starting_jobs
    kwargs['optimize_every_n_data'] = opt_frequency

    if 'trueM' in proc_name:
        kwargs['true_M'] = true_min_val

    if acq_dict['type'] == 'UCB':
        # lp_transform = 'softplus'
        lp_transform = 'none'
    else:
        lp_transform = 'none'

    if debug:
        kwargs['track_cond_k'] = False
        kwargs['create_plots'] = True
        kwargs['save_plots'] = True
        kwargs['plots_prefix'] = f"{proc_name}_{sampler.__name__}_"
        kwargs['verbose'] = 2

    if exp_type == 'async':
        bo = create_async_bo_instance(proc_name, sampler, surrogate, x_lim,
                                      lp_transform, async_interface, kwargs)
    elif exp_type == 'synch':
        bo = create_synch_bo_instance(proc_name, sampler, surrogate, x_lim,
                                      lp_transform, async_interface, kwargs)
    else:
        raise NotImplementedError

    bo.run()

    hash = get_commit_hash()

    # Combine the BO report with info about the experiment
    exp_desc = {
        'githash': hash,
        'seed': seed,
        'async_interface': async_interface.__class__.__name__,
        'proc_name': proc_name,
        'bo_strategy': f"{proc_name}_{acq_dict['type']}",
        'batch_size': batch_size,
        'n_workers': n_workers,
        'kern_choice': kern_choice,
        'model_opt_params': surrogate.opt_params,
        'func': sampler.__name__,
        'x_lim': x_lim,
        'optimise_surrogate_model': optimise_surrogate_model,
        'acq_opt_params': acq_opt_params,
        'acq_dict': acq_dict,
        'acq_type': acq_dict['type'],
        'y_min_opt_params': y_min_opt_params,
        'ard': ard_flag
    }

    exp_desc = pd.DataFrame([exp_desc])

    df_bo_exp = exp_desc.join(bo.df, how='right').ffill()

    if df_all_exps is None:
        df_all_exps = df_bo_exp
    else:
        df_all_exps = df_all_exps.append(df_bo_exp)

    if debug:
        bo.plot_y_min()
        plt.title(proc_name)
        plt.show()

    df_all_exps = df_all_exps.reset_index()

    print(f"Saving the exp summary to {save_fname}...")
    # df_all_exps.to_pickle(fname)
    time_limited_df_to_pickle(df_all_exps, save_fname, 20)
    print("Done!")
