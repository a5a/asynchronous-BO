# -*- coding: utf-8 -*-
"""
Main entry point for running BayesOpt experiments.

Defines high-level parameters that will be fed into bo_exp.

"""
from exps.defs import exp_async_synch
from exps.exp_utils import get_math_exp_task, \
    get_commit_hash, create_default_parser, \
    get_interface, get_iters, \
    generate_starting_data, create_intial_busy_jobs, \
    get_default_hp_priors_and_bounds
from ml_utils import timed_print as print

# *********   Parser   ******** #
parser = create_default_parser(synth_time=True)
parser.add_argument('-f', '--func', help='Function index.',
                    default=10, type=int)

args = parser.parse_args()
print(f"Got arguments: \n{args}")

# *********   Exp settings   ******** #
debug = False
force_run = False

foldername = get_commit_hash(with_date=True)

# *********   Overrides   ******** #
# args.func = 0
# args.proc = 'synch-LP'
# args.workers = 4
# args.batch = 4

# *********   Task and data   ******** #
f, x_lim, true_min_loc, true_min_val = get_math_exp_task(args.func)
hp_bounds, hyper_priors, restart_bounds = get_default_hp_priors_and_bounds(
    args, x_lim)
sampler = f

async_interface = get_interface(args.workers, synth=args.timer)

n_iter = get_iters(args, max_num_queries=200)

n_init = 3 * len(x_lim)
x_init, y_init = generate_starting_data(n_init, x_lim, sampler,
                                        async_interface=async_interface,
                                        seed=args.seed)

n_busy = args.workers
starting_jobs = create_intial_busy_jobs(n_busy, x_lim, sampler, args.seed)

# *********   Run   ******** #
exp_async_synch(args=args,
                sampler=sampler,
                true_min_val=true_min_val,
                x_init=x_init,
                y_init=y_init,
                x_lim=x_lim,
                foldername=foldername,
                hp_bounds=hp_bounds,
                restart_bounds=restart_bounds,
                hyper_priors=hyper_priors,
                n_iter=n_iter,
                starting_jobs=starting_jobs,
                async_interface=async_interface,
                force_run=force_run,
                debug=debug)
