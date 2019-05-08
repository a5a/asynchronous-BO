import argparse
import os
import subprocess

import GPy
import numpy as np

from bayesopt.async_bo import AsyncBOTS
from bayesopt.async_bo import PLAyBOOK_L, PLAyBOOK_H, PLAyBOOK_LL, PLAyBOOK_HL, \
    AsyncBOHeuristicQEI
from bayesopt.batch_bo import BatchBOLocalPenalisation, BatchBOHLP, BatchBOLLP, \
    BatchBOHLLP, BatchBOHeuristicQEI
from bayesopt.batch_bo import BatchBOTS
from bayesopt.executor import SimExecutorJumpToCompletedJob, \
    JobExecutorInSeriesBlocking
from ml_utils.math_functions import get_function


def create_synch_bo_instance(proc_name, sampler, surrogate, x_lim,
                             lp_transform, async_interface, kwargs):
    assert proc_name in (
        'synch-LP',
        'synch-HLP',
        'synch-LLP',
        'synch-HLLP',
        'synch-KB',
        'synch-CLmin',
        'synch-TS',
    )

    if proc_name == 'synch-LP':
        bo = BatchBOLocalPenalisation(sampler, surrogate, x_lim,
                                      lp_transform=lp_transform,
                                      async_interface=async_interface,
                                      **kwargs)

    elif proc_name == 'synch-HLP':
        bo = BatchBOHLP(
            sampler, surrogate, x_lim,
            async_interface=async_interface,
            lp_transform=lp_transform,
            **kwargs
        )

    elif proc_name == 'synch-LLP':
        bo = BatchBOLLP(
            sampler, surrogate, x_lim,
            async_interface=async_interface,
            lp_transform=lp_transform,
            **kwargs
        )

    elif proc_name == 'synch-HLLP':
        bo = BatchBOHLLP(
            sampler, surrogate, x_lim,
            async_interface=async_interface,
            lp_transform=lp_transform,
            **kwargs
        )

    elif proc_name == 'synch-KB':
        bo = BatchBOHeuristicQEI(sampler, surrogate, x_lim,
                                 async_interface=async_interface,
                                 **kwargs)
    elif proc_name == 'synch-CLmin':
        bo = BatchBOHeuristicQEI(sampler, surrogate, x_lim,
                                 async_interface=async_interface,
                                 **kwargs)

    elif proc_name == 'synch-TS':
        bo = BatchBOTS(
            sampler, surrogate, x_lim,
            async_interface=async_interface,
            **kwargs
        )
    else:
        raise NotImplementedError
    return bo


def create_async_bo_instance(proc_name, sampler, surrogate, x_lim,
                             lp_transform, async_interface, kwargs):
    assert proc_name in (
        'async-LP',
        'async-HLP',
        'async-LLP',
        'async-HLLP',
        'async-KB',
        'async-CLmin',
        'async-TS',
    )

    if proc_name == 'async-LP':
        bo = PLAyBOOK_L(sampler, surrogate, x_lim,
                        async_interface=async_interface,
                        lp_transform=lp_transform,
                        **kwargs)

    elif proc_name == 'async-HLP':
        bo = PLAyBOOK_H(
            sampler, surrogate, x_lim,
            async_interface=async_interface,
            lp_transform=lp_transform,
            **kwargs
        )


    elif proc_name == 'async-LLP':
        bo = PLAyBOOK_LL(sampler, surrogate, x_lim,
                         async_interface=async_interface,
                         lp_transform=lp_transform,
                         **kwargs)

    elif proc_name == 'async-HLLP':
        bo = PLAyBOOK_HL(
            sampler, surrogate, x_lim,
            async_interface=async_interface,
            lp_transform=lp_transform,
            **kwargs
        )

    elif proc_name == 'async-KB':
        bo = AsyncBOHeuristicQEI(sampler, surrogate, x_lim,
                                 async_interface=async_interface,
                                 **kwargs)
    elif proc_name == 'async-CLmin':
        bo = AsyncBOHeuristicQEI(sampler, surrogate, x_lim,
                                 async_interface=async_interface,
                                 **kwargs)
    elif proc_name == 'async-TS':
        bo = AsyncBOTS(
            sampler, surrogate, x_lim,
            async_interface=async_interface,
            **kwargs
        )
    else:
        raise NotImplementedError
    return bo


def get_math_exp_task(func, ard_flag=True):
    if func == 0:  # ackley
        f, x_lim, true_min_loc, true_min_val = get_function('ackley-3d')
    elif func == 1:  # branin
        f, x_lim, true_min_loc, true_min_val = get_function('branin-2d')
    elif func == 2:  # michalewicz 4d
        f, x_lim, true_min_loc, true_min_val = get_function('michalewicz-4d')
    elif func == 3:  # camelback
        f, x_lim, true_min_loc, true_min_val = get_function('camelback-2d')
    elif func == 4:  # rosenbrock
        f, x_lim, true_min_loc, true_min_val = get_function('rosenbrock-2d')
    elif func == 5:  # levy 5d
        f, x_lim, true_min_loc, true_min_val = get_function('levy-5d')
    elif func == 6:  # hartmann-6d
        f, x_lim, true_min_loc, true_min_val = get_function('hartmann-6d')
    elif func == 7:  # michalewicz 5d
        f, x_lim, true_min_loc, true_min_val = get_function('michalewicz-5d')
    elif func == 8:  # ackley 10d
        f, x_lim, true_min_loc, true_min_val = get_function('ackley-10d')
    elif func == 9:  # shekel 4d
        f, x_lim, true_min_loc, true_min_val = get_function('shekel-4d')
    elif func == 10:  # egg 2d
        f, x_lim, true_min_loc, true_min_val = get_function('egg-2d')
    elif func == 11:  # michalewicz 10d
        f, x_lim, true_min_loc, true_min_val = get_function('michalewicz-10d')
    elif func == 12:  # gsobol 5d
        f, x_lim, true_min_loc, true_min_val = get_function('gsobol-5d')
    elif func == 13:  # gsobol 10d
        f, x_lim, true_min_loc, true_min_val = get_function('gsobol-10d')
    elif func == 14:  # matern-2d
        f, x_lim, true_min_loc, true_min_val = get_function('matern-2d')
    elif func == 15:  # matern-6d
        f, x_lim, true_min_loc, true_min_val = get_function('matern-6d')
    elif func == 16:  # matern-10d
        f, x_lim, true_min_loc, true_min_val = get_function('matern-10d')
    elif func == 17:  # ackley 5d
        f, x_lim, true_min_loc, true_min_val = get_function('ackley-5d')
    else:
        raise NotImplementedError("Bad test function choice")

    return f, x_lim, true_min_loc, true_min_val


def get_default_hp_priors_and_bounds(args, x_lim):
    if args.ard:
        hp_bounds = np.array([[1e-6, 1000],  # kernel variance
                              *[[0.05, 1.]] * x_lim.shape[0],  # lengthscale
                              # [1e-6, 10]  # likelihood variance
                              ])

        # hyper_priors = [GPy.priors.Gamma(a=1.0, b=0.001),
        #                 # lengthscale prior keeps it small-ish
        #                 # *[GPy.priors.Uniform(1e-4, 5)] * x_lim.shape[0],
        #                 *[GPy.priors.Gamma(a=1.0, b=0.5)] * x_lim.shape[0],
        #                 # GPy.priors.Gamma(a=1.0, b=0.001)
        #                 ]

        hyper_priors = [GPy.priors.Gamma(a=1.0, b=0.001),
                        # lengthscale prior keeps it small-ish
                        # *[GPy.priors.Gamma(a=1.5, b=1.5)] * x_lim.shape[0],
                        *[GPy.priors.Gamma(a=1.0, b=0.01)] * x_lim.shape[0],
                        # GPy.priors.Gamma(a=1.0, b=0.001)
                        ]

        restart_bounds = hp_bounds
    else:
        hp_bounds = np.array([[1e-6, 1000],  # kernel variance
                              [1e-4, 5],  # lengthscale
                              # [1e-6, 1e6]  # likelihood variance
                              ])

        hyper_priors = [GPy.priors.Gamma(a=1.0, b=0.001),
                        # lengthscale prior keeps it small-ish
                        # GPy.priors.Uniform(1e-4, 5),
                        GPy.priors.Gamma(a=1.0, b=0.5),
                        # GPy.priors.Gamma(a=1.0, b=0.1)
                        ]

        restart_bounds = hp_bounds
    return hp_bounds, hyper_priors, restart_bounds


def get_synth_time_function(id):
    from scipy.stats import halfnorm
    if id == 0:  # positive normal
        def func():
            return halfnorm.rvs(scale=np.sqrt(np.pi / 2), size=1)
    else:
        raise NotImplementedError

    return func


def get_commit_hash(with_date=False):
    hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).strip()

    if with_date:
        commit_date = os.popen("git rev-list --format=format:'%ai' "
                               "--max-count=1 "
                               "`git rev-parse HEAD`").read().split('\n')[1] \
            .split()[0].replace('-', '')

        return str(commit_date) + '-' + str(hash.decode("utf-8")).strip()
    else:
        return hash


def create_default_parser(synth_time=False):
    """
    Creates basic exp input parser. Specific exps generally add extra params
    """
    parser = argparse.ArgumentParser(
        description="Run batch GP BayesOpt experiment")
    parser.add_argument('-s', '--seed', help='Set seed.',
                        default=0, type=int)
    parser.add_argument('-m', '--machine', help='Machine ID.',
                        default=0, type=int)
    parser.add_argument('-k', '--kernel', help='GP Kernel Choice. '
                                               '0 = RBF, 1 = Matern52',
                        default=1, type=int)
    parser.add_argument('-a', '--acquisition',
                        help='Choice of acquisition function. '
                             '0 = EI, 1 = PI, 2 = UCB, 3 = MES',
                        default=2, type=int)
    parser.add_argument('-to', '--tradeoff', help='Hyperparameter in acq func',
                        default=2., type=float)
    parser.add_argument('-p', '--proc', help='Async procedure. '
                                             'See code for options.',
                        default='async-HLLP', type=str)
    parser.add_argument('-b', '--batch', help='Batch size. Default = 1',
                        default=1, type=int)
    parser.add_argument('-w', '--workers', help='Number of parallel '
                                                'workers',
                        default=4, type=int)
    parser.add_argument('-od', '--optfreqdata',
                        help='After how many received data '
                             'points to optimize the surrogate',
                        default=4, type=int)
    parser.add_argument('-ard', '--ard', default=1, type=int,
                        help="Whether or not to use ARD. Default is true")

    if synth_time:
        parser.add_argument('-t', '--timer',
                            help='Which synth time function to '
                                 'use',
                            default=0, type=int)
    return parser


def get_interface(n_workers, synth=None):
    if synth is not None:
        synth_time_func = get_synth_time_function(synth)

        async_interface = \
            SimExecutorJumpToCompletedJob(n_workers=n_workers,
                                          time_func=synth_time_func)
    else:
        async_interface = \
            JobExecutorInSeriesBlocking(n_workers=n_workers)

    return async_interface


def apply_general_settings(args):
    seed = args.seed

    proc_name = args.proc
    if hasattr(args, 'noard'):
        ard_flag = not args.noard
    else:
        ard_flag = args.ard
    n_workers = args.workers
    np.random.seed(seed)

    # kernel
    if args.kernel == 0:
        kern_choice = 'rbf'
        kern_func = GPy.kern.RBF
    elif args.kernel == 1:
        kern_choice = 'matern52'
        kern_func = GPy.kern.Matern52
    else:
        raise NotImplementedError

    # acq
    if 'MES' in proc_name:
        args.acquisition = 3

    if args.acquisition == 0 or args.proc in (0, 1, 2):
        acq_dict = {'type': 'EI'}
    elif args.acquisition == 1:
        acq_dict = {'type': 'PI',
                    'tradeoff': args.tradeoff}
    elif args.acquisition == 2:
        acq_dict = {'type': 'UCB',
                    'tradeoff': args.tradeoff}
    elif args.acquisition == 3:
        acq_dict = {'type': 'MES',
                    'n_samples': 10}
    elif args.acquisition == 4:
        acq_dict = {'type': 'BALD'}
    elif args.acquisition == 5:
        acq_dict = {'type': 'EBALD'}
    else:
        raise NotImplementedError

    batch_size = args.batch

    opt_frequency = args.optfreqdata

    return seed, proc_name, batch_size, ard_flag, n_workers, kern_choice, \
           kern_func, acq_dict, opt_frequency


def get_iters(args, max_num_queries=200):
    batch_size = args.batch

    # Nguyen et al.,
    # Budgeted batch Bayesian optimization with unknown batch sizes
    n_iter = 100
    if n_iter * batch_size > max_num_queries:
        n_iter = int(max_num_queries / batch_size)

    return n_iter


def generate_starting_data(n_init, x_lim, sampler,
                           async_interface=None,
                           seed=None, task=None):
    save_flag = False
    init_fname = None

    np.random.seed(seed)
    # * Generate starting data
    x_init = np.random.uniform(x_lim[:, 0], x_lim[:, 1],
                               size=(n_init, x_lim.shape[0]))

    if async_interface is not None:
        jobs = []
        for ii in range(n_init):
            jobs.append({'x': x_init[ii], 'f': sampler})
        async_interface.add_job_to_queue(jobs)
        async_interface.run_until_empty()
        completed_jobs = async_interface.get_completed_jobs()
        x_init = np.vstack([j['x'] for j in completed_jobs])
        y_init = np.vstack([j['y'] for j in completed_jobs])
    else:
        y_init = np.zeros((len(x_init), 1))
        for ii, x in enumerate(x_init):
            y_init[ii] = sampler(x)

    if save_flag:
        np.savez(init_fname, x_init=x_init, y_init=y_init)

    return x_init, y_init


def create_intial_busy_jobs(n_busy, x_lim, sampler, seed):
    if n_busy > 0:
        starting_x_busy = np.random.uniform(x_lim[:, 0], x_lim[:, 1],
                                            size=(n_busy, x_lim.shape[0]))
        starting_jobs = [{'x': x, 'f': sampler} for x in
                         starting_x_busy]
    else:
        starting_jobs = None

    return starting_jobs


def get_default_optimisation_params(hp_bounds, restart_bounds,
                                    y_min=None):
    if y_min == 'samplegrad':
        y_min_opt_params = {'method': 'samplegrad',
                            'num_samples': 3000,
                            'num_local': 1,
                            'evaluate_sequentially': False,
                            'minimize_options': {'maxiter': 1000},
                            'verbose': False}
    else:
        y_min_opt_params = {'method': 'standard'}

    acq_opt_params = {'method': 'samplegrad',
                      'num_samples': 3000,
                      'num_local': 3,
                      'evaluate_sequentially': False,
                      'minimize_options': {'maxiter': 5000},
                      'verbose': False}
    min_acq_opt_params = {'method': 'samplegrad',
                          'num_samples': 3000,
                          'num_local': 0,
                          'evaluate_sequentially': False,
                          'verbose': False}
    gp_opt_params = {'method': 'multigrad',
                     'num_restarts': 10,
                     'restart_bounds': restart_bounds,
                     # likelihood variance
                     'hp_bounds': hp_bounds,
                     'verbose': False}
    optimise_surrogate_model = True

    return y_min_opt_params, acq_opt_params, min_acq_opt_params, \
           gp_opt_params, optimise_surrogate_model


def get_async_synch_from_args(args):
    exp = args.proc.split('-')[0]
    return exp


def get_strat_list(name):
    if name == 'ei':
        strats = [{'type': 'EI'}]
    elif name == 'ucb':
        strats = [{'type': 'UCB',
                   'tradeoff': 2.0}]
    elif name == 'bald':
        strats = [{'type': 'BALD'}]
    elif name == 'random-ucb':
        strats = [{'type': 'Random'},
                  {'type': 'UCB',
                   'tradeoff': 2.0}]
    elif name == 'random-ei':
        strats = [{'type': 'Random'},
                  {'type': 'EI'}]
    elif name == 'bald-ei':
        strats = [{'type': 'BALD'},
                  {'type': 'EI'}]
    elif name == 'bald-ucb':
        strats = [{'type': 'BALD'},
                  {'type': 'UCB',
                   'tradeoff': 2.0}]
    else:
        raise NotImplementedError
    return strats
