
This is the code for the design, evaluation and implementation of contextual bandits to learn optimal physical exercises regimes that decrease pain in endometriosis patients.

For more information, see our [arxiv submission](https://arxiv.org/abs/2309.14156).

## HTML Output
If you do not want to setup the python environment, `envaluation.html` still allows an interactive view of the intervention allocations in the browser.

## Setup
To run the jupyter notebook, you can install the necessary dependencies with:
```
python3 -m pip install -e .[dev]
```
Now, to start a jupyter lab, run
```
jupyter lab
```
We provide the used dataset from the paper under `data/2023-09-20-series.json`.

## Evaluation

The evaluation notebook can be found under `notebooks/evaluation.ipynb`.

## Data Generation
The data was generated using a custom simulation library.
You can view the setup of the bayesian and data generating models in pymc under `notebooks/data-generation.ipynb`.
