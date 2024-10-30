# gpfilter.py

**`gpfilter`** implements the gaussian process (GP) as discrete time state space model to be implemented into bayesian filters (BF).

The filter implementations are based on 
[`filterpy`](https://github.com/rlabbe/filterpy) 
and
[`torchfilter`](https://github.com/stanford-iprl-lab/torchfilter)
.
The `torchfilter` library is augmented with an interacting multiple model particle filter (IMM-PF) which is based on the existing particle filter (pf) implementation and the algorithm presented in [1].
The gaussian process models are from [`GpyTorch`](https://github.com/cornellius-gp/gpytorch).


TBC.


### Installation

```bash
$ git clone https://github.com/adrianLepp/gpfilter.py
$ cd gpfilter.py
$ pip install -e .
```

### References

[1] A. Lepp und D. Weidemann,
"[Interacting-Multiple-Model Partikelfilter zur Fehleridentifikation](https://www.asim-gi.org/fileadmin/user_upload_asim/ASIM_Publikationen_OA/AM180/a2024.arep.20_OA.pdf)",
in: ASIM 2022, Jan. 2022, S. 187–194. doi: 10.11128/arep.20.a2024. url:
a2024.arep.20 OA.pdf

