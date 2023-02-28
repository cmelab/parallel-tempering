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
    parameters["n_density"] = [0.8]
    parameters["n_particles"] = [100]
    parameters["r"] = [0.5]
    parameters["r_cut"] = [2.5]
    parameters["energy_func"] = ["lj"]
    parameters["hard_sphere"] = [False]

    # LJ energy parameters
    parameters["epsilon"] = [1.0]
    parameters["sigma"] = [0.5]
    parameters["n"] = [12]
    parameters["m"] = [6]
    parameters["e_factor"] = [0.01, 0.1, 0.2, 0.5, 0.7,  1.]

    # logging parameters
    parameters["energy_write_freq"] = [500]
    parameters["trajectory_write_freq"] = [5000]

    # run parameters
    parameters["mixing_steps"] = [1e6]
    parameters["mixing_kT"] = [10]
    parameters["mixing_max_trans"] = [0.5]

    parameters["n_steps"] = [[5e5]]
    parameters["kT"] = [[1.5]]
    parameters["max_trans"] = [[0.4]]
    parameters["seed"] = [20]

    return list(parameters.keys()), list(product(*parameters.values()))


custom_job_doc = {}  # add keys and values for each job document created


def main(root):
    project = signac.init_project("MCMC-project", root=root)  # Set the signac project name
    param_names, param_combinations = get_parameters()
    # Create jobs
    for params in param_combinations:
        parent_statepoint = dict(zip(param_names, params))
        parent_job = project.open_job(parent_statepoint)
        parent_job.init()
        parent_job.doc.setdefault("job_type", "sim")
        parent_job.doc.setdefault("swap", False)
        parent_job.doc.setdefault("done", False)
        parent_job.doc.setdefault("current_run", 0)
        parent_job.doc.setdefault("mixed", False)
        parent_job.doc.setdefault("timestep", [])
        parent_job.doc.setdefault("accepted_moves", [])
        parent_job.doc.setdefault("rejected_moves", [])
        parent_job.doc.setdefault("acceptance_ratio", [])
        parent_job.doc.setdefault("tps", [])
        parent_job.doc.setdefault("energy", [])
        parent_job.doc.setdefault("avg_PE", [])
    if custom_job_doc:
        for key in custom_job_doc:
            parent_job.doc.setdefault(key, custom_job_doc[key])

    project.write_statepoints()

def init_jobs(root='../'):
    logging.basicConfig(level=logging.INFO)
    main(root)


if __name__ == "__main__":
    init_jobs()
