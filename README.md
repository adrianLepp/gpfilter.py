# gpfilter.py

**`gpfilter`** implements the gaussian process (GP) as discrete time state space model for hybrid state estimation with the interacting multiple model (IMM).
The bayes filter (BF) implementations are based on 
[`filterpy`](https://github.com/rlabbe/filterpy) (Unscented Kalman Filter) 
and
[`torchfilter`](https://github.com/stanford-iprl-lab/torchfilter) (Particle Filter)
.
Both libraries implement also other filters, which should be able to used here, but are not testet yet.
The `torchfilter` library is augmented with an interacting multiple model particle filter (IMM-PF) which is based on the existing particle filter implementation and the IMM-PF algorithm presented in [1].
Gaussian process state space models models are derived from [`GpyTorch`](https://github.com/cornellius-gp/gpytorch).

The implementation based on `torchfilter` is to be favored, since everythin is based on `pytorch` and the parameter optimization could be done for the whole filter module at once. At the moment optimization is only done for the single gp models, but in future this should be implemented since it makes much more sence to tune the gp predictions according to the filter performance. 

### Installation

```bash
$ git clone https://github.com/adrianLepp/gpfilter.py
$ cd gpfilter.py
$ pip install -e .
```

### Use

To see how to use the models, check the files in the submodule `examples`, where one implentation for torchfilter and filterpy is done for a nonlinear water tank system. 

### References

[1] A. Lepp und D. Weidemann,
"[Interacting-Multiple-Model Partikelfilter zur Fehleridentifikation](https://www.asim-gi.org/fileadmin/user_upload_asim/ASIM_Publikationen_OA/AM180/a2024.arep.20_OA.pdf)",
in: ASIM 2022, Jan. 2022, S. 187–194. doi: 10.11128/arep.20.a2024. url:
a2024.arep.20 OA.pdf

