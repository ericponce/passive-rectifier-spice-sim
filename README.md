# passive-rectifier-spice-sim
Python and LTSpice based passive rectifier simulation 

## File Description
* `Makefile` is used by `sim.py` to accomplish its job. 
* `incremental_impedance.py` uses `sim.py` to run LTSpice on your choice of `base_<model>.asc` which depends on `base_<model>.gen` (which is generated by `incremental_impedance.py`)

Currently, two models are supported:
* RLC Model, where the L is in series with a parallel R and C
* PFC CPL Model, using a simplified but continuous conduction model of a bandwidth limited power factor corrected constant power load.

## System Configuration
Start by setting up a virtual environment:
```
mkdir env
python3 -m venv env
```

Then, activate it and install required packages
```
source env/bin/activate
pip install -r requirements.txt
```

## Project Configuration
* Set `LTSPICE` variable in `Makefile`

## Incremental Imepdance Simulation

To run: `python incremental_impedance.py`