#!/usr/bin/env python3

from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive
import torch
from torch.nn import Module
from torch import Tensor
from typing import Optional

def fix_nonpositive_definite(a: Tensor) -> Tensor:
    EPS = 1e-6;
    ZERO = 1e-10;
    [eigenval,eigenvec] = torch.linalg.eig(a)
    
    for n in range(len(eigenval)):
        if eigenval[n].abs() <= ZERO:
            eigenval[n] = EPS

    sigma = eigenvec * torch.diag(eigenval) * eigenvec.t()

    return sigma

def inv_transform(x):
    return x


def is_psd(mat):
    return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real>=0).all())

class PositiveDefinite(Module):
    def __init__(self, eps=1e-6, zero=1e-10):
        """
        TODO
        """
        dtype = torch.get_default_dtype()
        eps = torch.as_tensor(eps).to(dtype)
        zero = torch.as_tensor(zero).to(dtype)

        super().__init__()

        self.register_buffer("eps", eps)
        self.register_buffer("zero", zero)

        self._transform = fix_nonpositive_definite
        self._inv_transform = inv_transform

        # if transform is not None and inv_transform is None:
        #     self._inv_transform = _get_inv_param_transform(transform)

        # if initial_value is not None:
        #     self._initial_value = self.inverse_transform(torch.as_tensor(initial_value))
        # else:
        self._initial_value = None

    def _apply(self, fn):
        self.eps = fn(self.eps)
        self.zero = fn(self.zero)
        return super()._apply(fn)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        result = super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=False,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )
        # The lower_bound and upper_bound buffers are new, and so may not be present in older state dicts
        # Because of this, we won't have strict-mode on when loading this module
        return result

    @property
    def enforced(self) -> bool:
        return self._transform is not None

    def check(self, tensor) -> bool:
        return is_psd(tensor)

    def check_raw(self, tensor) -> bool:
        return is_psd(self.transform(tensor))

    def transform(self, tensor: Tensor) -> Tensor:
        """
        TODO
        """
        if not self.enforced:
            return tensor

        n = tensor.shape[0]
        transformed_tensor = torch.zeros_like(tensor)
        for i in range(n):
            transformed_tensor[i] = self._transform(tensor[i])
        #transformed_tensor = self._transform(tensor)

        return transformed_tensor

    def inverse_transform(self, transformed_tensor: Tensor) -> Tensor:
        """
        Applies the inverse transformation.
        """
        if not self.enforced:
            return transformed_tensor

        tensor = self._inv_transform(transformed_tensor)

        return tensor

    @property
    def initial_value(self) -> Optional[Tensor]:
        """
        The initial parameter value (if specified, None otherwise)
        """
        return self._initial_value

    def __repr__(self) -> str:
        if self.eps.numel() == 1 and self.zero.numel() == 1:
            return self._get_name() + f"({self.eps:.3E}, {self.zero:.3E})"
        else:
            return super().__repr__()

    def __iter__(self):
        yield self.eps
        yield self.zero

class ConvolvedProcessKernel(Kernel):
    r"""
    :param int num_tasks: Number of tasks
    :param int num_latents: Number of latent processes
    :param dict kwargs: Additional arguments to pass to the kernel.
    """

    is_stationary = True

    def __init__(
        self,
        num_tasks: int,
        num_latents: int,
        num_inputs: int,
        **kwargs,
    ):
        """"""
        super().__init__(**kwargs)

        self.num_tasks = num_tasks
        self.num_latents = num_latents
        self.num_inputs = num_inputs

        # register hyperparameters

        # varianceCoefficient
        self.register_parameter(
            name='raw_varianceCoefficient', parameter=torch.nn.Parameter(torch.ones(self.num_tasks, self.num_latents, 1))
        )
        # # set the parameter constraint to be positive, when nothing is specified
        # if varianceCoefficient_constraint is None:
        #     varianceCoefficient_constraint = Positive()

        # # register the constraint
        self.register_constraint("raw_varianceCoefficient", Positive())

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        # if varianceCoefficient_prior is not None:
        #     self.register_prior(
        #         "varianceCoefficient_prior",
        #         varianceCoefficient_prior,
        #         lambda m: m.varianceCoefficient,
        #         lambda m, v : m._set_varianceCoefficient(v),
        #     )
        # Create a tensor consisting of num_matrices stacked 3x3 identity matrices

        # Precision matrix for outputs
        self.register_parameter(
            name='raw_output_precisionMatrix', parameter=torch.nn.Parameter(torch.stack([torch.eye(self.num_inputs)] * self.num_tasks))
        )

        self.register_constraint("raw_output_precisionMatrix", Positive())

        # Precision matrix for latent processes
        self.register_parameter(
            name='raw_latent_precisionMatrix', parameter=torch.nn.Parameter(torch.stack([torch.eye(self.num_inputs)] * self.num_latents))
        )

        self.register_constraint("raw_latent_precisionMatrix", Positive())

    #varianceCoefficient
    @property
    def varianceCoefficient(self):
        return self.raw_varianceCoefficient_constraint.transform(self.raw_varianceCoefficient)

    @varianceCoefficient.setter
    def varianceCoefficient(self, value):
        return self._set_varianceCoefficient(value)

    def _set_varianceCoefficient(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_varianceCoefficient)
        self.initialize(raw_varianceCoefficient=self.raw_varianceCoefficient_constraint.inverse_transform(value))
    
    #Precision matrix for outputs
    @property
    def output_precisionMatrix(self):
        return self.raw_output_precisionMatrix_constraint.transform(self.raw_output_precisionMatrix)

    @output_precisionMatrix.setter
    def output_precisionMatrix(self, value):
        return self._set_output_precisionMatrix(value)

    def _set_output_precisionMatrix(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_output_precisionMatrix)
        self.initialize(raw_output_precisionMatrix=self.raw_output_precisionMatrix_constraint.inverse_transform(value))

    #Precision matrix for latent
    @property
    def latent_precisionMatrix(self):
        return self.raw_latent_precisionMatrix_constraint.transform(self.raw_latent_precisionMatrix)

    @latent_precisionMatrix.setter
    def latent_precisionMatrix(self, value):
        return self._set_latent_precisionMatrix(value)

    def _set_latent_precisionMatrix(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_latent_precisionMatrix)
        self.initialize(raw_latent_precisionMatrix=self.raw_latent_precisionMatrix_constraint.inverse_transform(value))

    #--------------------------------------------
    def forward_deprecated(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("ConvolutionProcessKernel does not accept the last_dim_is_batch argument.")
        
        n = x1.size(0)
        m = x2.size(0)
        covar = torch.zeros((n*self.num_tasks, m*self.num_tasks))

        for i in range(n):
            xi = x1[i].unsqueeze(-1)
            for j in range(m):
                xj = x2[j].unsqueeze(-1)
                for d in range(self.num_tasks):
                    for dp in range(self.num_tasks):
                        c = 0
                        for q in range(self.num_latents):
                            P_eqv = torch.inverse(self.output_precisionMatrix[d]) + torch.inverse(self.output_precisionMatrix[dp]) + torch.inverse(self.latent_precisionMatrix[q]) 
                            diff = xi - xj
                            c1 = (self.varianceCoefficient[d,q]* self.varianceCoefficient[dp,q]) / ( torch.pow(2 * torch.pi, torch.tensor(self.num_inputs/2)) * torch.pow(torch.norm(P_eqv), 1/2) )
                            #c1 = ( torch.pow(2 * torch.pi, torch.tensor(self.num_inputs/2)) * torch.pow(torch.norm(P_eqv), 1/2) )
                            c2 = torch.exp(- 1/2 *  torch.mm(torch.mm(diff.transpose(0,1),torch.inverse(P_eqv) ),diff))
                            c += c1 * c2
                        covar[i*self.num_tasks + d, j*self.num_tasks + dp] = c
        return covar

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks


    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("ConvolutionProcessKernel does not accept the last_dim_is_batch argument.")
        
        n = x1.size(0)
        m = x2.size(0)
        xa = x1.repeat_interleave(self.num_tasks, dim  =0 ).view(-1,1,1,self.num_inputs).repeat(1,m*self.num_tasks,1,1)
        xb = x2.repeat_interleave(self.num_tasks, dim  =0 ).view(1,-1,1, self.num_inputs).repeat(n*self.num_tasks,1,1,1)
        diff = xa - xb

        covar = torch.zeros((n*self.num_tasks, m*self.num_tasks))

        for q in range(self.num_latents):
            S = self.varianceCoefficient.squeeze()

            SSt = S[:,q:q+1].mm(S[:,q:q+1].t())
            PInv = self.output_precisionMatrix.inverse() # m x num_tasks x num_tasks
            AInv = self.latent_precisionMatrix[q].inverse() # num_tasks x num_tasks
            Peqv1 = torch.stack([PInv]* self.num_tasks, dim=0)
            Peqv2 = torch.stack([PInv]* self.num_tasks, dim=1)
            Peqv3 = torch.stack( [torch.stack([AInv]* self.num_tasks, dim=0)]* self.num_tasks, dim=0)
            assert Peqv1.shape == Peqv2.shape == Peqv3.shape == (self.num_tasks, self.num_tasks, self.num_inputs, self.num_inputs)
            
            Peqv = Peqv1 + Peqv2 + Peqv3
            Peqv_abs = Peqv.norm(dim=(2,3))
            c1 = (SSt / ( torch.pow(2 * torch.pi, torch.tensor(self.num_inputs/2)) * torch.pow(Peqv_abs, 1/2) )) #/ num_latents x num_tasks
            #c1 = ( torch.pow(2 * torch.pi, torch.tensor(self.num_inputs/2)) * torch.pow(Peqv_abs, 1/2) )
            assert c1.shape == (self.num_tasks, self.num_tasks)


            Peqv_stack = Peqv.inverse().repeat(n, m , 1, 1) # n * num_tasks x m * num_tasks x num_inputs x num_inputs

            
            c2 = torch.exp(- 1/2 *  diff @ Peqv_stack @ diff.transpose(-1,-2)).squeeze()
            c1_stack = c1.repeat(n, m)
            c = c1_stack * c2
            covar += c


            # for i in range(n):
            #     xi = x1[i].unsqueeze(0)
            #     for j in range(m):
            #         xj = x2[j].unsqueeze(0)
            #         diff = xi - xj # N x num_inputs
            #         c2 = torch.exp(- 1/2 *  diff @ Peqv.inverse() @ diff.t()).squeeze()
            #         assert c2.shape == (self.num_tasks, self.num_tasks)

            #         c = c1 * c2
            #         assert c.shape == (self.num_tasks, self.num_tasks)

            #         covar[i*self.num_tasks:i*self.num_tasks+self.num_tasks , j*self.num_tasks:j*self.num_tasks+self.num_tasks] += c
        retVal = covar.diagonal(dim1=-1, dim2=-2) if diag else covar
        return retVal

        return covar
