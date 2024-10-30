from kernel import ConvolvedProcessKernel
import torch


num_tasks = 3
num_latents = 2
num_inputs = 4

output_precisionMatrix = torch.stack([torch.eye(num_inputs)] * num_tasks)
output_precisionMatrix[1,:,:] +=torch.eye(num_inputs)
output_precisionMatrix[2,:,:] +=torch.eye(num_inputs)*2

S =  torch.arange(0, num_tasks * num_latents).reshape(num_tasks, num_latents)

latent_precisionMatrix = torch.stack([torch.eye(num_inputs)] * num_latents)
varianceCoefficient = torch.ones(num_tasks, num_latents)

kernel = ConvolvedProcessKernel(num_tasks=num_tasks, num_latents=num_latents, num_inputs=num_inputs)

def kernelFct(x1, x2):
    for d in range(num_tasks):
        for dp in range(num_tasks):
            c = 0
            for q in range(num_latents):
                P_eqv = torch.inverse(output_precisionMatrix[d]) + torch.inverse(output_precisionMatrix[dp]) + torch.inverse(latent_precisionMatrix[q]) 
                diff = x1 - x2 # N x num_inputs

                # this stays the same for all inputs
                c1 = (varianceCoefficient[d,q]* varianceCoefficient[dp,q]) / ( torch.pow(2 * torch.pi, torch.tensor(num_inputs/2)) * torch.pow(torch.norm(P_eqv), 1/2) )
                c2 = torch.exp(- 1/2 *  torch.mm(torch.mm(diff.transpose(0,1),torch.inverse(P_eqv) ),diff))
                c += c1 * c2
            covar = c
    return covar

def kernelFct2(x1, x2):
    
    for q in range(num_latents):

        SSt = S[:,q:q+1].mm(S[:,q:q+1].t())
        PInv = output_precisionMatrix.inverse() # m x num_tasks x num_tasks
        AInv = latent_precisionMatrix[q].inverse() # num_tasks x num_tasks
        Peqv1 = torch.stack([PInv]* num_tasks, dim=0)
        Peqv2 = torch.stack([PInv]* num_tasks, dim=1)
        Peqv3 = torch.stack( [torch.stack([AInv]* num_tasks, dim=0)]* num_tasks, dim=0)
        assert Peqv1.shape == Peqv2.shape == Peqv3.shape == (num_tasks, num_tasks, num_inputs, num_inputs)
        
        Peqv = Peqv1 + Peqv2 + Peqv3
        Peqv_abs = Peqv.norm(dim=(2,3))
        c1 = (SSt / ( torch.pow(2 * torch.pi, torch.tensor(num_inputs/2)) * torch.pow(Peqv_abs, 1/2) )) #/ num_latents x num_tasks
        assert c1.shape == (num_tasks, num_tasks)

        #print(c1)

        n = x1.size(0)
        m = x2.size(0)
        covar = torch.zeros((n*num_tasks, m*num_tasks))

        for i in range(n):
            xi = x1[i].unsqueeze(0)
            for j in range(m):
                xj = x2[j].unsqueeze(0)
                diff = xi - xj # N x num_inputs
                c2 = torch.exp(- 1/2 *  diff @ Peqv.inverse() @ diff.t()).squeeze()
                assert c2.shape == (num_tasks, num_tasks)

                c = c1 * c2
                assert c.shape == (num_tasks, num_tasks)

                covar[i*num_tasks:i*num_tasks+num_tasks , j*num_tasks:j*num_tasks+num_tasks] += c

        return covar

x1 = torch.ones(2,4)
x2 = torch.ones(2,4)

x = torch.arange(1,3).reshape(2,1) * torch.ones(1,4)

c = kernel.forward_deprecated(x1,x2)
c2 = kernel(x1, x2)

print( (c-c2).numpy())


# st = S.unsqueeze(-1) * S.t().unsqueeze(0) # num_tasks x num_latents x num_tasks
# SSt = S.mm(S.t()) #  num_tasks x num_tasks

# PInv = output_precisionMatrix.inverse() # m x num_tasks x num_tasks
# AInv = latent_precisionMatrix.inverse() # q x num_tasks x num_tasks

# #PInv_stack = PInv.unsqueeze(1).expand(-1, num_latents, -1, -1)

# PInv_stack = torch.stack([PInv] * num_latents, dim=1)
# AInv_stack = torch.stack([AInv] * num_tasks, dim=0)
# Peqv = PInv_stack * 2 + AInv_stack

# Peqv_abs = Peqv.norm(dim=(2,3)) 




# print(PInv_stack.shape)

# #res = kernel(x, x)


# print(Peqv.numpy())






