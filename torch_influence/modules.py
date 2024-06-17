import logging
from typing import Callable, Optional

import numpy as np
import scipy.sparse.linalg as L

import cupy as cp
import cupyx.scipy.sparse.linalg as L_gpu

import torch
from torch import nn
from torch.utils import data
import gpytorch
from gpytorch.lazy import LazyTensor

from torch_influence.base import BaseInfluenceModule, BaseObjective


class AutogradInfluenceModule(BaseInfluenceModule):
    r"""An influence module that computes inverse-Hessian vector products
    by directly forming and inverting the risk Hessian matrix using :mod:`torch.autograd`
    utilities.

    Args:
        model: the model of interest.
        objective: an implementation of :class:`BaseObjective`.
        train_loader: a training dataset loader.
        test_loader: a test dataset loader.
        device: the device on which operations are performed.
        damp: the damping strength :math:`\lambda`. Influence functions assume that the
            risk Hessian :math:`\mathbf{H}` is positive definite, which often fails to
            hold for neural networks. Hence, a damped risk Hessian :math:`\mathbf{H} + \lambda\mathbf{I}`
            is used instead, for some sufficiently large :math:`\lambda > 0` and
            identity matrix :math:`\mathbf{I}`.
        check_eigvals: if ``True``, this initializer checks that the damped risk Hessian
            is positive definite, and raises a :mod:`ValueError` if it is not. Otherwise,
            no check is performed.
        store_as_hessian: if ``True``, damped risk Hessian is stores in object and any access to
            inverse Hessian will be lazy. Otherwise, inverse Hessian is computed immediately.

    Warnings:
        This module scales poorly with the number of model parameters :math:`d`. In
        general, computing the Hessian matrix takes :math:`\mathcal{O}(nd^2)` time and
        inverting it takes :math:`\mathcal{O}(d^3)` time, where :math:`n` is the size
        of the training dataset.
    """

    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            damp: float,
            check_eigvals: bool = False,
            store_as_hessian: bool = False
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        self.store_as_hessian = store_as_hessian
        self.damp = damp

        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        d = flat_params.shape[0]
        hess = 0.0

        for batch, batch_size in self._loader_wrapper(train=True):
            def f(theta_):
                self._model_reinsert_params(self._reshape_like_params(theta_))
                return self.objective.train_loss(self.model, theta_, batch)

            hess_batch = torch.autograd.functional.hessian(f, flat_params).detach().cpu()
            hess = hess + hess_batch * batch_size

        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)
            hess = hess / len(self.train_loader.dataset)
            hess = hess + damp * torch.eye(d, device=hess.device)

            if check_eigvals:
                eigvals = torch.linalg.eigvalsh(hess.cpu()).numpy()
                logging.info("hessian min eigval %f", np.min(eigvals).item())
                logging.info("hessian max eigval %f", np.max(eigvals).item())
                if not bool(np.all(eigvals >= 0)):
                    raise ValueError()

            if self.store_as_hessian:
                self.hess = hess
                self.inverse_hess = None
            else:
                self.inverse_hess = torch.inverse(hess)

    def get_hessian(self):
        if not self.store_as_hessian:
            raise NotImplementedError("To access damped risk Hessian, set store_as_hessian to True.")
        return self.hess

    def get_inverse_hessian(self):
        if self.inverse_hess is None:
            # Lazy inverse computation
            self.inverse_hess = torch.inverse(self.hess)
        return self.inverse_hess

    def inverse_hvp(self, vec):
        inverse_hess = self.get_inverse_hessian()
        return inverse_hess @ vec


class CGInfluenceModule(BaseInfluenceModule):
    r"""An influence module that computes inverse-Hessian vector products
    using the method of (truncated) Conjugate Gradients (CG).

    This module relies :func:`scipy.sparse.linalg.cg()` to perform CG.

    Args:
        model: the model of interest.
        objective: an implementation of :class:`BaseObjective`.
        train_loader: a training dataset loader.
        test_loader: a test dataset loader.
        device: the device on which operations are performed.
        damp: the damping strength :math:`\lambda`. Influence functions assume that the
            risk Hessian :math:`\mathbf{H}` is positive-definite, which often fails to
            hold for neural networks. Hence, a damped risk Hessian :math:`\mathbf{H} + \lambda\mathbf{I}`
            is used instead, for some sufficiently large :math:`\lambda > 0` and
            identity matrix :math:`\mathbf{I}`.
        gnh: if ``True``, the risk Hessian :math:`\mathbf{H}` is approximated with
            the Gauss-Newton Hessian, which is positive semi-definite.
            Otherwise, the risk Hessian is used.
        **kwargs: keyword arguments which are passed into the "Other Parameters" of
            :func:`scipy.sparse.linalg.cg()`.
    """

    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            damp: float,
            gnh: bool = False,
            use_cupy: bool = True,
            **kwargs
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        self.damp = damp
        self.gnh = gnh
        self.use_cupy = use_cupy
        self.cg_kwargs = kwargs

    def inverse_hvp(self, vec):
        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        # Know which device tensors should sit on
        if self.use_cupy:
            if self.device == "cuda":
                device_id = 0
            else:
                device_id = int(self.device.split(":")[-1])

        def hvp_fn(v):
            # v_ = v.clone().detach().requires_grad_(False)
            if self.use_cupy:
                v_ = torch.as_tensor(v, device=self.device)
            else:
                v_ = torch.tensor(v, requires_grad=False, device=self.device, dtype=vec.dtype)
            """
            was_squeezed = len(v_.shape) > 1 and v_.shape[1] == 1
            if was_squeezed:
                v_ = v_.squeeze(1)
            """

            hvp = 0.0
            for batch, batch_size in self._loader_wrapper(train=True):
                hvp_batch = self._hvp_at_batch(batch, flat_params, vec=v_, gnh=self.gnh)
                hvp = hvp + hvp_batch.detach() * batch_size
            hvp = hvp / len(self.train_loader.dataset)
            hvp = hvp + self.damp * v_

            """
            if was_squeezed:
                hvp = hvp.unsqueeze(1)
            """

            if self.use_cupy:
                # Wrap as cupy array, but create a copy
                # return cp.asarray(hvp.cpu().numpy())
                # Get ID that self.device corresponds to
                with cp.cuda.Device(device_id):
                    return cp.asarray(hvp.cpu().numpy())
            return hvp.cpu().numpy()

        d = vec.shape[0]
        if self.use_cupy:
            """
            # Wrap hvp_fn in a LinearOperator-like class
            class HvpLazyTensor(LazyTensor):
                def __init__(self, hvp_fn, vec_shape):
                    self.hvp_fn = hvp_fn
                    self.vec_shape = vec_shape
                    super(HvpLazyTensor, self).__init__(torch.eye(vec_shape, dtype=vec.dtype))

                def _matmul(self, rhs):
                    return self.hvp_fn(rhs)

                def _transpose_nonbatch(self):
                    return self  # Symmetric Hessian

                def _size(self):
                    return torch.Size((self.vec_shape, self.vec_shape))

            # Create the lazy tensor
            hvp_lazy_tensor = HvpLazyTensor(hvp_fn, d)

            # rtol: 1e-5
            # default niters: len(g) * 10

            # Convert vec to a tensor if it isn't already
            vec_tensor = vec if torch.is_tensor(vec) else torch.tensor(vec, dtype=vec.dtype)
            print(vec_tensor.shape)

            # Solve using conjugate gradients
            ihvp = gpytorch.utils.linear_cg(hvp_fn, vec_tensor, **self.cg_kwargs)
            """

            with cp.cuda.Device(device_id):
                linop = L_gpu.LinearOperator((d, d), matvec=hvp_fn)
                vec_ = cp.asarray(vec)
                ihvp  = L_gpu.cg(A=linop, b=vec_, **self.cg_kwargs)[0]
        else:
            # Slow Scipy-based method
            linop = L.LinearOperator((d, d), matvec=hvp_fn)
            ihvp = L.cg(A=linop, b=vec.cpu().numpy(), **self.cg_kwargs)[0]

        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)

        # if self.use_cupy:
            # return torch.as_tensor(ihvp, device=self.device)
        return torch.tensor(ihvp, device=self.device)


class LiSSAInfluenceModule(BaseInfluenceModule):
    r"""An influence module that computes inverse-Hessian vector products
    using the Linear time Stochastic Second-Order Algorithm (LiSSA).

    At a high level, LiSSA estimates an inverse-Hessian vector product
    by using truncated Neumann iterations:

    .. math::
        \mathbf{H}^{-1}\mathbf{v} \approx \frac{1}{R}\sum\limits_{r = 1}^R
        \left(\sigma^{-1}\sum_{t = 1}^{T}(\mathbf{I} - \sigma^{-1}\mathbf{H}_{r, t})^t\mathbf{v}\right)

    Here, :math:`\mathbf{H}` is the risk Hessian matrix and :math:`\mathbf{H}_{r, t}` are
    loss Hessian matrices over batches of training data drawn randomly with replacement (we
    also use a batch size in ``train_loader``). In addition, :math:`\sigma > 0` is a scaling
    factor chosen sufficiently large such that :math:`\sigma^{-1} \mathbf{H} \preceq \mathbf{I}`.

    In practice, we can compute each inner sum recursively. Starting with
    :math:`\mathbf{h}_{r, 0} = \mathbf{v}`, we can iteratively update for :math:`T` steps:

    .. math::
        \mathbf{h}_{r, t} = \mathbf{v} + \mathbf{h}_{r, t - 1} - \sigma^{-1}\mathbf{H}_{r, t}\mathbf{h}_{r, t - 1}

    where :math:`\mathbf{h}_{r, T}` will be equal to the :math:`r`-th inner sum.

    Args:
        model: the model of interest.
        objective: an implementation of :class:`BaseObjective`.
        train_loader: a training dataset loader.
        test_loader: a test dataset loader.
        device: the device on which operations are performed.
        damp: the damping strength :math:`\lambda`. Influence functions assume that the
            risk Hessian :math:`\mathbf{H}` is positive-definite, which often fails to
            hold for neural networks. Hence, a damped risk Hessian :math:`\mathbf{H} + \lambda\mathbf{I}`
            is used instead, for some sufficiently large :math:`\lambda > 0` and
            identity matrix :math:`\mathbf{I}`.
        repeat: the number of trials :math:`R`.
        depth: the recurrence depth :math:`T`.
        scale: the scaling factor :math:`\sigma`.
        gnh: if ``True``, the risk Hessian :math:`\mathbf{H}` is approximated with
            the Gauss-Newton Hessian, which is positive semi-definite.
            Otherwise, the risk Hessian is used.
        debug_callback: a callback function which is passed in :math:`(r, t, \mathbf{h}_{r, t})`
            at each recurrence step.
     """

    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            damp: float,
            repeat: int,
            depth: int,
            scale: float,
            gnh: bool = False,
            debug_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None
    ):

        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        self.damp = damp
        self.gnh = gnh
        self.repeat = repeat
        self.depth = depth
        self.scale = scale
        self.debug_callback = debug_callback

    def inverse_hvp(self, vec):

        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        ihvp = 0.0

        for r in range(self.repeat):

            h_est = vec.clone()

            for t, (batch, _) in enumerate(self._loader_wrapper(sample_n_batches=self.depth, train=True)):

                hvp_batch = self._hvp_at_batch(batch, flat_params, vec=h_est, gnh=self.gnh)

                with torch.no_grad():
                    hvp_batch = hvp_batch + self.damp * h_est
                    h_est = vec + h_est - hvp_batch / self.scale

                if self.debug_callback is not None:
                    self.debug_callback(r, t, h_est)

            ihvp = ihvp + h_est / self.scale

        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)

        return ihvp / self.repeat


class HVPModule(BaseInfluenceModule):
    r"""Basic module for easily computing HPV

    """
    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            device: torch.device
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=None,
            device=device,
        )

    #TODO: Allow user to specify dtype for higher precision

    def hvp(self, vec):
        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        hvp = 0.0
        for batch, batch_size in self._loader_wrapper(train=True):
            hvp_batch = self._hvp_at_batch(batch, flat_params, vec=vec.to(self.device), gnh=False)
            hvp = hvp + hvp_batch.detach() * batch_size
        hvp = hvp / len(self.train_loader.dataset)

        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)

        return hvp

    def inverse_hvp(self, vec):
        raise NotImplementedError("Inverse HVP not implemented for HVPModule - this is a heler class for HVP")


class ShanksSablonniereModule(BaseInfluenceModule):
    """
    Based on Series of Hessian-Vector Products for Tractable Saddle-Free Newton Optimisation of Neural Networks (https://arxiv.org/abs/2310.14901).
    Compute recursive Shanks transformation of given sequence using Samelson inverse and the epsilon-algorithm with a Sablonniere modifier.
    """
    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            device: torch.device,
            # damp: float,
            **kwargs
    ):
        self.acceleration_order = kwargs.get('acceleration_order', 8)
        self.initial_scale_factor = kwargs.get('initial_scale_factor', 100)
        self.num_update_steps = kwargs.get('num_update_steps', 20)

        self.hvp_module = HVPModule(
            model,
            objective,
            train_loader,
            device=device
        )
        # TODO: Consider adding damping

    def compute_epsilon_acceleration(
        self,
        source_sequence,
        num_applications: int=1,):
        """Compute `num_applications` recursive Shanks transformation of
        `source_sequence` (preferring later elements) using `Samelson` inverse and the
        epsilon-algorithm, with Sablonniere modifier.
        """

        def inverse(vector):
            """
            Samelson inverse
            """
            return vector / vector.dot(vector)

        epsilon = {}
        for m, source_m in enumerate(source_sequence):
            epsilon[m, 0] = source_m.squeeze(1)
            epsilon[m + 1, -1] = 0

        s = 1
        m = (len(source_sequence) - 1) - 2 * num_applications
        initial_m = m
        while m < len(source_sequence) - 1:
            while m >= initial_m:
                # Sablonniere modifier
                inverse_scaling = np.floor(s / 2) + 1

                epsilon[m, s] = epsilon[m + 1, s - 2] + inverse_scaling * inverse(
                    epsilon[m + 1, s - 1] - epsilon[m, s - 1]
                )
                epsilon.pop((m + 1, s - 2))
                m -= 1
                s += 1
            m += 1
            s -= 1
            epsilon.pop((m, s - 1))
            m = initial_m + s
            s = 1

        return epsilon[initial_m, 2 * num_applications]

    def inverse_hvp(self, vec):
        # Detach and clone input
        vector_cache = vec.detach().clone()
        update_sum   = vec.detach().clone()
        coefficient_cache = 1

        cached_update_sums = []
        if self.acceleration_order > 0 and self.num_update_steps == 2 * self.acceleration_order + 1:
            cached_update_sums.append(update_sum)

        # Do HessianSeries calculation
        for update_step in range(1, self.num_update_steps):
            hessian2_vector_cache = self.hvp_module.hvp(self.hvp_module.hvp(vector_cache))

            if update_step == 1:
                scale_factor = torch.norm(hessian2_vector_cache, p=2) / torch.norm(vec, p=2)
                scale_factor = max(scale_factor.item(), self.initial_scale_factor)

            vector_cache = (vector_cache - (1/scale_factor)*hessian2_vector_cache).clone()
            coefficient_cache *= (2 * update_step - 1) / (2 * update_step)
            update_sum += coefficient_cache * vector_cache

            if self.acceleration_order > 0 and update_step >= (self.num_update_steps - 2 * self.acceleration_order - 1):
                cached_update_sums.append(update_sum.clone())

        # Perform series acceleration (Shanks acceleration)
        if self.acceleration_order > 0:
            accelerated_sum = self.compute_epsilon_acceleration(
                cached_update_sums, num_applications=self.acceleration_order
            )
            accelerated_sum /= np.sqrt(scale_factor)
            accelerated_sum = accelerated_sum.unsqueeze(1)
        
            return accelerated_sum

        update_sum /= np.sqrt(scale_factor)
        return update_sum
