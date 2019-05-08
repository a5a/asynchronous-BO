# -*- coding: utf-8 -*-

"""
Acquisition functions
"""
from typing import Optional, List

import numpy as np
from scipy.stats import norm

from ml_utils.models import GP


class AcquisitionFunction(object):
    """
    Base class for acquisition functions. Used to define the interface
    """

    def __init__(self, surrogate=None, verbose=False):
        self.surrogate = surrogate
        self.verbose = verbose

    def evaluate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError


class AcquisitionWithOffset(AcquisitionFunction):
    """
    Offset is *subtracted* from the acquisition value
    """

    def __init__(self, acq, offset=None, verbose=None):
        self.acq = acq
        if offset is not None:
            self.offset = offset
        super().__init__(verbose=verbose)

    def __str__(self) -> str:
        return f"Offset-{self.acq}"

    def evaluate(self, x: np.ndarray, **kwargs):
        # adding 1e-3 for numerical stability
        return self.acq.evaluate(x).flatten() - self.offset + 1e-3


class EI(AcquisitionFunction):
    """
    Expected improvement acquisition function for a Gaussian model

    Model should return (mu, var)
    """

    def __init__(self, surrogate: GP, best: np.ndarray, verbose=False):
        self.best = best
        super().__init__(surrogate, verbose)

    def __str__(self) -> str:
        return "EI"

    def evaluate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluates the EI acquisition function.

        Parameters
        ----------
        x
            Input to evaluate the acquisition function at

        """
        if self.verbose:
            print("Evaluating EI at", x)
        mu, var = self.surrogate.predict(np.atleast_2d(x))
        var = np.clip(var, 1e-8, np.inf)

        s = np.sqrt(var)
        gamma = (self.best - mu) / s
        return (s * gamma * norm.cdf(gamma) + s * norm.pdf(gamma)).flatten()


class Uncertainty(AcquisitionFunction):
    """ Uncertainty acquisition function returns the variance of the surrogate
    """

    def __init__(self, surrogate: GP, verbose=False):
        super().__init__(surrogate, verbose)

    def __str__(self) -> str:
        return "Uncertainty"

    def evaluate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluates the acquisition function.

        Parameters
        ----------
        x
            Input to evaluate the acquisition function at

        """
        if self.verbose:
            print("Evaluating Uncertainty at", x)
        _, var = self.surrogate.predict(x)
        var = np.clip(var, 1e-8, np.inf)

        return var


class PI(AcquisitionFunction):
    """
    Probability of improvement acquisition function for a Gaussian model

    Model should return (mu, var)
    """

    def __init__(self, surrogate: GP, best: np.ndarray, tradeoff: float,
                 verbose=False):
        self.best = best
        self.tradeoff = tradeoff

        super().__init__(surrogate, verbose)

    def __str__(self) -> str:
        return f"PI-{self.tradeoff}"

    def evaluate(self, x, **kwargs) -> np.ndarray:
        """
        Evaluates the PI acquisition function.

        Parameters
        ----------
        x
            Input to evaluate the acquisition function at

        """
        if self.verbose:
            print("Evaluating PI at", x)
        mu, var = self.surrogate.predict(x)
        var = np.clip(var, 1e-8, np.inf)

        s = np.sqrt(var)
        gamma = (self.best - mu - self.tradeoff) / s
        return norm.cdf(gamma).flatten()


class UCB(AcquisitionFunction):
    """
    Upper confidence bound acquisition function for a Gaussian model

    Model should return (mu, var)
    """

    def __init__(self, surrogate: GP, tradeoff: float, verbose=False):
        self.tradeoff = tradeoff

        super().__init__(surrogate, verbose)

    def __str__(self) -> str:
        return f"UCB-{self.tradeoff}"

    def evaluate(self, x, **kwargs) -> np.ndarray:
        """
        Evaluates the UCB acquisition function.

        Parameters
        ----------
        x
            Input to evaluate the acquisition function at
        """
        if self.verbose:
            print("Evaluating UCB at", x)
        mu, var = self.surrogate.predict(x)
        var = np.clip(var, 1e-8, np.inf)

        s = np.sqrt(var)  # type: np.ndarray
        return -(mu - self.tradeoff * s).flatten()


class PenalisedAcquisition(AcquisitionFunction):
    """Penalised acquisition function parent class

    Parameters
    ----------
    surrogate

    acq
        An instance of AcquisitionFunction, e.g. EI, UCB, etc

    x_batch
        Locations already in the batch

    best
        Best value so far of the function

    verbose
    """

    def __init__(self, surrogate: GP,
                 acq: AcquisitionFunction,
                 x_batch: np.ndarray,
                 best: Optional[np.ndarray] = None,
                 verbose=False):
        super().__init__(surrogate, verbose)

        if best is None:
            try:
                self.best = acq.best
            except NameError:
                self.best = None
            except AttributeError:
                self.best = None
        else:
            self.best = best

        # shape is (1 x n_samples), or float
        if isinstance(best, np.ndarray):
            self.best = self.best.reshape(1, -1)

        self.acq = acq
        self.x_batch = x_batch

    def __str__(self) -> str:
        return f"{self.acq.__str__}-LP{len(self.x_batch)}"

    def evaluate(self, x, **kwargs) -> np.ndarray:
        """Evaluate the penalised acquisition function.

        Note that the result is log(acq), as this makes adding in the
        penalizers easier and numerically more stable. The resulting
        location of the optimum remains the same

        Parameters
        ----------
        x
            Location(s) to evaluate the acquisition function at

        Returns
        -------
        np.ndarray
            Value(s) of the acquisition function at x
        """
        out = self._penalized_acquisition(x)
        # if np.sum(np.isnan(out)) > 0:
        #     print(f"penalised acq is nan at {x[np.where(np.isnan(out))]}")
        return out

    def _penalized_acquisition(self, x):
        raise NotImplementedError


class LocallyPenalisedAcquisition(PenalisedAcquisition):
    """LP Acquisition function for use in Batch BO via Local Penalization

    Parameters
    ----------
    surrogate

    acq
        An instance of AcquisitionFunction, e.g. EI, UCB, etc

    L
        Estimate of the Lipschitz constant

    x_batch
        Locations already in the batch

    best
        Best value so far of the function

    transform
        None or softplus

    verbose
    """

    def __init__(self, surrogate: GP,
                 acq: AcquisitionFunction,
                 x_batch: np.ndarray,
                 L,
                 best: Optional[np.ndarray] = None,
                 transform='softplus',
                 verbose=False):

        super().__init__(surrogate,
                         acq,
                         x_batch,
                         best=best,
                         verbose=verbose)

        self.L = L

        if transform is None:
            self.transform = 'none'
        else:
            self.transform = transform

        self.r_x0, self.s_x0 = self._hammer_function_precompute()

    def _hammer_function_precompute(self):
        """
        Pre-computes the parameters of a penalizer centered at x0.
        """
        x0 = self.x_batch
        best = self.best
        surrogate = self.surrogate
        L = self.L

        assert x0 is not None

        if len(x0.shape) == 1:
            x0 = x0[None, :]
        m = surrogate.predict(x0)[0]
        pred = surrogate.predict(x0)[1].copy()
        pred[pred < 1e-16] = 1e-16
        s = np.sqrt(pred)
        r_x0 = np.abs(m - best) / L
        s_x0 = s / L
        r_x0 = r_x0.flatten()
        s_x0 = s_x0.flatten()
        return r_x0, s_x0

    def _hammer_function(self, x, x0, r, s):
        '''
        Creates the function to define the exclusion zones
        '''
        return norm.logcdf((np.sqrt(
            (np.square(
                np.atleast_2d(x)[:, None, :] -
                np.atleast_2d(x0)[None, :, :])).sum(-1)) - r) / s)

    def _penalized_acquisition(self, x):
        '''
        Creates a penalized acquisition function using 'hammer' functions
        around the points collected in the batch
        .. Note:: the penalized acquisition is always mapped to the log
        space. This way gradients can be computed additively and are more
        stable.
        '''
        fval = self.acq.evaluate(x)
        x_batch = self.x_batch
        r_x0 = self.r_x0
        s_x0 = self.s_x0

        if self.transform == 'softplus':
            fval_org = fval.copy()
            fval = np.log1p(np.exp(fval_org))
        elif self.transform == 'none':
            fval = fval + 1e-50

        if x_batch is not None:
            log_fval = np.log(fval)
            h_vals = self._hammer_function(x, x_batch, r_x0, s_x0)
            log_fval += h_vals.sum(axis=-1)
            fval = np.exp(log_fval)
        return fval


class LocalLipschitzPenalisedAcquisition(LocallyPenalisedAcquisition):
    """LLP Acquisition function for use in Batch BO via Local Penalization
    with local Lipschitz constants

    Parameters
    ----------
    surrogate

    acq
        An instance of AcquisitionFunction, e.g. EI, UCB, etc

    L
        Estimates of the Lipschitz constant at each batch point

    x_batch
        Locations already in the batch

    best
        Best value so far of the function

    transform
        None or softplus

    verbose
    """

    def __init__(self, surrogate: GP,
                 acq: AcquisitionFunction,
                 x_batch: np.ndarray,
                 L: np.ndarray,
                 best: Optional[np.ndarray] = None,
                 transform='softplus',
                 verbose=False):
        super().__init__(surrogate,
                         acq,
                         x_batch,
                         L,
                         best=best,
                         transform=transform,
                         verbose=verbose)

    def _hammer_function_precompute(self):
        """
        Pre-computes the parameters of a penalizer centered at x0.
        """
        x0 = self.x_batch
        best = self.best
        surrogate = self.surrogate
        L = self.L

        assert x0 is not None

        if len(x0.shape) == 1:
            x0 = x0[None, :]
        m = surrogate.predict(x0)[0].flatten()
        pred = surrogate.predict(x0)[1].copy().flatten()
        pred[pred < 1e-16] = 1e-16
        s = np.sqrt(pred)
        r_x0 = np.abs(m - best) / L
        # try:
        #     r_x0 = np.abs(m - best) / L
        # except ValueError as e:
        #     print(f"Failed!\nm = {m}\nbest = {best}\nL = {L}")
        #     sys.exit()

        s_x0 = s / L
        r_x0 = r_x0.flatten()
        s_x0 = s_x0.flatten()
        return r_x0, s_x0

    def _hammer_function(self, x, x0, r, s):
        '''
        Creates the function to define the exclusion zones
        '''
        return norm.logcdf((np.sqrt(
            (np.square(
                np.atleast_2d(x)[:, None, :] -
                np.atleast_2d(x0)[None, :, :])).sum(-1)) - r) / s)


class HardMinAwareConeAcquisition(PenalisedAcquisition):
    """HLP Acquisition function for use in Batch BO

    Cone with information on y_min

    Parameters
    ----------
    surrogate

    acq
        An instance of AcquisitionFunction, e.g. EI, UCB, etc

    L
        Estimate of the Lipschitz constant

    x_batch
        Locations already in the batch

    best
        Best value so far of the function

    transform
        None or softplus

    verbose
    """

    def __init__(self, surrogate: GP,
                 acq: AcquisitionFunction,
                 x_batch: np.ndarray,
                 L,
                 best: Optional[np.ndarray] = None,
                 transform='softplus',
                 verbose=False,
                 **kwargs):

        super().__init__(surrogate,
                         acq,
                         x_batch,
                         best=best,
                         verbose=verbose)

        self.L = L

        if transform is None:
            self.transform = 'none'
        else:
            self.transform = transform

        self.r_mu, self.r_std = self._cone_function_precompute()

    def _cone_function_precompute(self):
        x0 = self.x_batch
        L = self.L
        M = self.best
        mu, var = self.surrogate.predict(x0)
        r_mu = (mu.flatten() - M) / L
        r_std = np.sqrt(var.flatten()) / L

        r_mu = r_mu.flatten()
        r_std = r_std.flatten()
        return r_mu, r_std

    def _cone_function(self, x, x0):
        """
        Creates the function to define the exclusion zones

        Using half the Lipschitz constant as the gradient of the penalizer.

        We use the log of the penalizer so that we can sum instead of multiply
        at a later stage.
        """
        # L = self.L
        # M = self.best
        # mu, var = self.surrogate.predict(x0)
        # r_mu = (mu - M) / L
        # r_std = np.sqrt(var) / L
        #
        # r_mu = r_mu.flatten()
        # r_std = r_std.flatten()
        r_mu = self.r_mu
        r_std = self.r_std

        x_norm = np.sqrt(np.square(
            np.atleast_2d(x)[:, None, :] - np.atleast_2d(x0)[None, :, :]).sum(
            -1))
        norm_jitter = 0  # 1e-100
        # return 1 / (r_mu + r_std).reshape(-1, len(x0)) * (x_norm + norm_jitter)
        return 1 / (r_mu + r_std) * (x_norm + norm_jitter)

    def _penalized_acquisition(self, x):
        '''
        Creates a penalized acquisition function using the 4th norm between
        the acquisition function and the cone
        '''
        fval = self.acq.evaluate(x)
        x_batch = self.x_batch

        if self.transform == 'softplus':
            fval_org = fval.copy()
            fval = np.log1p(np.exp(fval_org))
        elif self.transform == 'none':
            fval = fval + 1e-50

        if x_batch is not None:
            h_vals = self._cone_function(x, x_batch).prod(-1)
            h_vals = h_vals.reshape([1, -1])
            clipped_h_vals = np.linalg.norm(
                np.concatenate((h_vals,
                                np.ones(h_vals.shape)), axis=0), -5,
                axis=0)

            fval *= clipped_h_vals

        return fval
