import pdb
from typing import Tuple

import numpy as np


def find_between(val, func, funcvals, mgrid, thres):
    t2 = np.argmin(abs(funcvals - val))
    check_diff = abs(funcvals[0, t2] - val)
    if abs(funcvals[0, t2] - val) < thres:
        res = mgrid[t2]
        return res

    assert funcvals[0, 0] < val and funcvals[0, -1] > val
    if funcvals[0, t2] > val:
        left = mgrid[t2 - 1]
        right = mgrid[t2]
    else:
        left = mgrid[t2]
        right = mgrid[t2 + 1]

    mid = (left + right) / 2.0
    midval = func(mid)
    cnt = 1
    while abs(midval - val) > thres:
        if midval > val:
            right = mid
        else:
            left = mid

        mid = (left + right) / 2
        midval = func(mid)
        cnt = cnt + 1
        if cnt > 10000:
            pdb.set_trace()
    res = mid
    return res


def add_hallucinations_to_x_and_y(bo, old_x, old_y, x_new, fixed_dim_vals=None) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Add hallucinations to the data arrays.

    Parameters
    ----------
    old_x
        Current x values
    old_y
        Current y values
    x_new
        Locations at which to use the async infill procedure. If x_busy
        is None, then nothing happens and the x and y arrays are returned

    Returns
    -------
    augmented_x (np.ndarray), augmented_y (list or np.ndarray)
    """
    if x_new is None:
        x_out = old_x
        y_out = old_y
    else:
        if isinstance(x_new, list):
            x_new = np.vstack(x_new)

        if fixed_dim_vals is not None:
            if fixed_dim_vals.ndim == 1:  # row vector
                fixed_dim_vals = np.vstack([fixed_dim_vals] * len(x_new))
            assert len(fixed_dim_vals) == len(x_new)
            x_new = np.hstack((
                fixed_dim_vals, x_new
            ))
        x_out = np.vstack((old_x, x_new))
        fake_y = make_hallucinated_data(bo, x_new, bo.async_infill_strategy)
        y_out = np.vstack((old_y, fake_y))

    return x_out, y_out


def make_hallucinated_data(bo, x: np.ndarray,
                           strat: str) -> np.ndarray:
    """Returns fake y-values based on the chosen heuristic

    Parameters
    ----------
    x
        Used to get the value for the kriging believer. Otherwise, this
        sets the number of values returned

    bo
        Instance of BayesianOptimization

    strat
        string describing the type of hallucinated data. Choices are:
        'constant_liar_min', 'constant_liar_median', 'kriging_believer',
        'posterior_simple'

    Returns
    -------
    y : np.ndarray
        Values for the desired heuristic

    """
    if strat == 'constant_liar_min':
        if x is None:
            y = np.atleast_2d(bo.y_min)
        else:
            y = np.array([bo.y_min] * len(x)).reshape(-1, 1)

    elif strat == 'constant_liar_median':
        if x is None:
            y = np.atleast_2d(bo.y_min)
        else:
            y = np.array([bo.y_min] * len(x)).reshape(-1, 1)

    elif strat == 'kriging_believer':
        y = bo.surrogate.predict(x)[0]

    elif strat == 'posterior_simple':
        mu, var = bo.surrogate.predict(x)
        y = np.random.multivariate_normal(mu.flatten(),
                                          np.diag(var.flatten())) \
            .reshape(-1, 1)

    elif strat == 'posterior_full':
        mu, var = bo.surrogate.predict(x, full_cov=True)
        y = np.random.multivariate_normal(mu.flatten(), var).reshape(-1, 1)
    else:
        raise NotImplementedError

    return y


def stable_cholesky(M):
    """ Returns L, a 'stable' cholesky decomposition of M. L is lower triangular and
        satisfies L*L' = M.
        Sometimes nominally psd matrices are not psd due to numerical issues. By adding a
        small value to the diagonal we can make it psd. This is what this function does.
        Use this iff you know that K should be psd. We do not check for errors
    """
    # pylint: disable=superfluous-parens
    if M.size == 0:
        return M  # if you pass an empty array then just return it.
    try:
        # First try taking the Cholesky decomposition.
        L = np.linalg.cholesky(M)
    except np.linalg.linalg.LinAlgError:
        # If it doesn't work, then try adding diagonal noise.
        diag_noise_power = -11
        max_M = np.diag(M).max()
        diag_noise = np.diag(M).max() * 1e-11
        chol_decomp_succ = False
        while not chol_decomp_succ:
            try:
                L = np.linalg.cholesky(
                    M + (10 ** diag_noise_power * max_M) * np.eye(M.shape[0]))
                chol_decomp_succ = True
            except np.linalg.linalg.LinAlgError:
                diag_noise_power += 1
        if diag_noise_power >= 5:
            print('**************** Cholesky failed: Added diag noise = %e' % (
                diag_noise))
    return L
