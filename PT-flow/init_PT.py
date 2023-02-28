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

    # simulation mode: MCMC-flow or polymer-flow
    parameters["mode"] = ["MCMC-flow"]
    parameters["seed"] = [20]

    # total number of swaps
    parameters["n_attempts"] = [10]
    # initial wait time (including mixing run)
    parameters["first_wait"] = [800]
    # initial wait time (in seconds) before checking the status of jobs
    parameters["init_wait"] = [400]
    # additional wait time (in seconds) after the first wait time is over and jobs are still not done
    parameters["extra_wait"] = [100]
    # maximum wait time (in seconds)
    parameters["max_tries"] = [5]

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
        parent_job.doc.setdefault("job_type", "PT")
        parent_job.doc.setdefault("current_attempt", 0)
        parent_job.doc.setdefault("swap_history", [])
        parent_job.doc.setdefault("accepted_attempts", [])

    project.write_statepoints()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()