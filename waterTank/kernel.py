#!/usr/bin/env python3

from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive
import torch

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
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
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


