#!/usr/bin/env python
"""Initialize the project's data space.
Iterates over all defined state points and initializes
the associated job workspace directories.
The result of running this file is the creation of a signac workspace:
    - signac.rc file containing the project name
    - signac_statepoints.json summary for the entire workspace
    - workspace/ directory that contains a sub-directory of every individual statepoint
    - signac_statepoints.json within each individual statepoint sub-directory.
"""

import logging
from collections import OrderedDict
from itertools import product

import signac


def get_parameters():
    parameters = OrderedDict()

    # system parameters
    # Put your density state points here:
    parameters["mode"] = ["MCMC-flow"]
    parameters["workspace"] = ["../MCMC-flow/"]
    parameters["n_attempts"] = [10]
    parameters["PT_n_steps"] = [[1e6]]
    parameters["PT_kT"] = [[1.5]]
    parameters["seed"] = [20]

    return list(parameters.keys()), list(product(*parameters.values()))


def main():
    project = signac.init_project("parallel-tempering")  # Set the signac project name
    param_names, param_combinations = get_parameters()
    # Create jobs
    for params in param_combinations:
        parent_statepoint = dict(zip(param_names, params))
        parent_job = project.open_job(parent_statepoint)
        parent_job.init()
        parent_job.doc.setdefault("done", False)
        parent_job.doc.setdefault("current_attempt", 0)
        parent_job.doc.setdefault("swap_history", [])
        parent_job.doc.setdefault("accepted_attempts", [])

    project.write_statepoints()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()