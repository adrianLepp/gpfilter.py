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

        # register the constraint
        self.register_constraint("raw_varianceCoefficient", Positive())

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
            PInv = self.output_precisionMatrix.inverse()
            AInv = self.latent_precisionMatrix[q].inverse()
            Peqv1 = torch.stack([PInv]* self.num_tasks, dim=0)
            Peqv2 = torch.stack([PInv]* self.num_tasks, dim=1)
            Peqv3 = torch.stack( [torch.stack([AInv]* self.num_tasks, dim=0)]* self.num_tasks, dim=0)
            assert Peqv1.shape == Peqv2.shape == Peqv3.shape == (self.num_tasks, self.num_tasks, self.num_inputs, self.num_inputs)
            
            Peqv = Peqv1 + Peqv2 + Peqv3
            Peqv_abs = Peqv.norm(dim=(2,3))
            c1 = (SSt / ( torch.pow(2 * torch.pi, torch.tensor(self.num_inputs/2)) * torch.pow(Peqv_abs, 1/2) ))
            assert c1.shape == (self.num_tasks, self.num_tasks)

            Peqv_stack = Peqv.inverse().repeat(n, m , 1, 1)
            
            c2 = torch.exp(- 1/2 *  diff @ Peqv_stack @ diff.transpose(-1,-2)).squeeze()
            c1_stack = c1.repeat(n, m)
            c = c1_stack * c2
            covar += c

        retVal = covar.diagonal(dim1=-1, dim2=-2) if diag else covar
        return retVal
