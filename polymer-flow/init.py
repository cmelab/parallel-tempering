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
    parameters["molecule"] = ["PPS"]
    parameters["system"] = ["Pack"]
    parameters["density"] = [1.28]
    parameters["n_chains"] = [[30]]
    parameters["polymer_lengths"] = [15]
    parameters["forcefield"] = ["OPLS_AA_PPS"]
    parameters["remove_hydrogens"] = [True]
    parameters["seed"] = [20]

    # Sim parameters
    parameters["r_cut"] = [2.5]
    parameters["dt"] = [0.0003]
    parameters["e_factor"] = [1.0]

    # run parameters
    parameters["shrink_steps"] = [5e6]
    parameters["shrink_kT"] = [4.0]
    parameters["shrink_period"] = [1000]
    parameters["n_steps"] = [1e7]
    parameters["kT"] = [
            2.0,
            2.2,
            2.4,
            2.6,
            2.8,
            3.0,
            3.2,
            3.4,
            3.6,
            3.8,
            4.0,
            4.2,
            4.4,
            4.6,
    ]
    parameters["tau_kt"] = [0.01]

    # logging parameters
    parameters["num_gsd_frames"] = [400]
    parameters["num_data_logs"] = [50000]

    return list(parameters.keys()), list(product(*parameters.values()))



def main(root):
    project = signac.init_project("poly-flow", root=root)
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
        parent_job.doc.setdefault("ran_shrink", False)
        parent_job.doc.setdefault("timestep", [])
        parent_job.doc.setdefault("tps", [])
        parent_job.doc.setdefault("energy", [])
        parent_job.doc.setdefault("avg_PE", [])
        parent_job.doc.total_steps = int(
                parent_job.sp.n_steps + parent_job.sp.shrink_steps
        )

    project.write_statepoints()

def init_jobs(root='../'):
    logging.basicConfig(level=logging.INFO)
    main(root)


if __name__ == "__main__":
    init_jobs()
