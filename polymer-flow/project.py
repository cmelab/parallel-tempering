"""Define the project's workflow logic and operation functions.
Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:
    $ python src/project.py --help
"""
import os
import shutil
import sys
import pickle
import numpy as np
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
import gsd.hoomd
import hoomd_polymers
import hoomd_polymers.molecules
import hoomd_polymers.systems
import hoomd_polymers.forcefields
from hoomd_polymers.sim import Simulation


class MyProject(FlowProject):
    pass


class Borah(DefaultSlurmEnvironment):
    hostname_pattern = r"borah|.*\.cm\.cluster"
    template = "borah.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="short",
            help="Specify the partition to submit to."
        )


class R2(DefaultSlurmEnvironment):
    hostname_pattern = "r2"
    template = "r2.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="shortq",
            help="Specify the partition to submit to."
        )


class Fry(DefaultSlurmEnvironment):
    hostname_pattern = r"fry|node*"
    template = "fry.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="batch",
            help="Specify the partition to submit to."
        )


# Definition of project-related labels (classification)
@MyProject.label
def sampled(job):
    return job.doc.get("done")

@MyProject.label
def averaged(job):
    return job.doc.get("averaged")

@MyProject.label
def initialized(job):
    return job.isfile("trajectory.gsd")


@MyProject.label
def is_sim(job):
    return job.doc["job_type"] == "sim"

@MyProject.label
def pt_done(job):
    return job.doc.get("pt_done")


# Useful functions
def load_pickle_ff(job, ff_file):
    """Load list of hoomd forces from pickle files in a job's workspace."""
    with open(job.fn(ff_file), "rb") as f:
        hoomd_ff = pickle.load(f)
    return hoomd_ff


@directives(executable="python -u")
@directives(ngpu=1)
@MyProject.operation
@MyProject.post(sampled)
@MyProject.pre(is_sim)
def sample(job):
    with job:
        print("-----------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("-----------------------")
        print("----------------------")
        print("Creating the system...")
        print("----------------------")

        # Setting up the system
        if job.isfile("restart.gsd"): # Initializing from a restart.gsd
            with gsd.hoomd.open(job.fn("restart.gsd")) as traj:
                init_snap = traj[0]
            hoomd_ff = load_pickle_ff(job, "forcefield.pickle")
        else: # No restart, generate the system and apply a FF
            molecule_obj = getattr(hoomd_polymers.molecules, job.sp.molecule)
            ff_obj = getattr(hoomd_polymers.forcefields, job.sp.forcefield)
            system_obj = getattr(hoomd_polymers.systems, job.sp.system)
            mol_kwargs = {"length": job.sp.polymer_lengths}
            system = system_obj(
                    molecule=molecule_obj,
                    n_mols=job.sp.n_chains,
                    density=job.sp.density,
                    mol_kwargs=mol_kwargs,
                    packing_expand_factor=3,
            )
            print("----------------------")
            print("System generated...")
            print("----------------------")
            print("Applying the forcefield...")
            print("----------------------")
            system.apply_forcefield(
                    forcefield=ff_obj(),
                    remove_hydrogens=job.sp.remove_hydrogens,
                    remove_charges=True
            )
            init_snap = system.hoomd_snapshot
            hoomd_ff = system.hoomd_forcefield

            job.doc.gsd_write_frequency = int(
                    job.doc.total_steps / job.sp.num_gsd_frames
            )
            job.doc.log_write_frequency = int(
                    job.doc.total_steps / job.sp.num_data_logs
            )

        # Set up stuff to initialize a Simulation
        sim = Simulation(
                initial_state=init_snap,
                forcefield=hoomd_ff,
                gsd_write_freq=job.doc.gsd_write_frequency,
                log_write_freq=job.doc.log_write_frequency,
        )
        if not job.isfile("forcefield.pickle"): # Pickle FF for future runs
            sim.pickle_forcefield(file_path=job.fn("forcefield.pickle"))
        if job.sp.e_factor != 1:
            print("Scaling LJ epsilon values...")
            sim.adjust_epsilon(scale_by=job.sp.e_factor)

        print("----------------------")
        print("Starting simulation...")
        print("----------------------")
        # Run shrink simulation
        if not job.doc.ran_shrink:
            job.doc.target_box_reduced = (
                    system.target_box*10/system.reference_distance
            )
            sim.run_update_volume(
                    final_box_lengths=job.doc.target_box_reduced,
                    n_steps=job.sp.shrink_steps,
                    period=int(job.sp.shrink_period),
                    kT=job.sp.shrink_kT,
                    tau_kt=job.sp.tau_kt
            )
            assert np.array_equal(
                    sim.box_lengths_reduced, job.doc.target_box_reduced
            )
            job.doc.ran_shrink = True
            print("----------------------")
            print("Shrink simulation finished...")
            print("----------------------")

        # Run NVT simulation. # Starting here for jobs after swaps
        sim.run_NVT(
                n_steps=job.sp.n_steps, kT=job.sp.kT, tau_kt=job.sp.tau_kt
        )
        sim.save_restart_gsd()
        print("----------------------")
        print("Simulation finished...")
        print("----------------------")

        if job.doc["swap"]:
            shutil.copyfile(
                    job.fn("trajectory.gsd"),
                    job.fn(f"trajectory_{job.doc.current_run}_swap.gsd")
            )
            shutil.copyfile(
                    job.fn("sim_data.txt"),
                    job.fn(f"sim_data_{job.doc.current_run}_swap.txt")
            )
        else:
            shutil.copyfile(
                    job.fn("trajectory.gsd"),
                    job.fn(f"trajectory_{job.doc.current_run}.gsd")
            )
            shutil.copyfile(
                    job.fn("sim_data.txt"),
                    job.fn(f"sim_data_{job.doc.current_run}.txt")
            )

        job.doc["timestep"].append(sim.timestep)
        job.doc["current_run"] += 1
        job.doc["done"] = True


@directives(executable="python -u")
@directives(ngpu=0)
@MyProject.operation
@MyProject.post(averaged)
@MyProject.pre(pt_done)
def variables(job):
    import numpy as np
    from cmeutils.structure import (
            radius_of_gyration, end_to_end, nematic_order_param
    )

    # Rg
    window_rgs = [] # means
    window_rg_stds = [] # stds
    window_rg_arrays = [] # list of vals
    # Re
    window_res = []
    window_re_stds = []
    window_re_arrays = []
    # S2 Order Param
    window_order_param = []

    potential_energy = []

    for i in range(1, 102):
        gsd_file = job.fn(f"trajectory_{i}.gsd")
        rg_mean, rg_std, rg_array = radius_of_gyration(
                gsd_file=str(gsd_file), start=0, stop=-1
        )
        window_rgs.append(rg_mean)
        window_rg_stds.append(rg_std)
        window_rg_arrays.append(rg_array)

        re_array, re_mean, re_stds, re_vectors = end_to_end(gsd_file, 4, 102, 0, -1)
        window_res.append(re_mean)
        window_re_stds.append(re_std)
        window_re_arrays.append(re_array)

        for vec in re_vectors:
            op = nematic_order_param(vec, director=(1, 1, 1))
            window_order_param.append(op.order)

        log_file = job.fn(f"sim_data_{i}.txt")
        pe = np.genfromtxt(log_file, names=True)[
                "md.compute.ThermodynamicQuantities.potential_energy"
        ]
        potential_energy.extend(pe.tolist())

    np.save(file=job.fn("rg_mean.npy"), arr=np.asarray(window_rgs))
    np.save(file=job.fn("rg_std.npy"), arr=np.asarray(window_rg_stds))
    np.save(file=job.fn("rg_array.npy"), arr=np.asarray(window_rg_arrays))
    np.save(file=job.fn("re_mean.npy"), arr=np.asarray(window_res))
    np.save(file=job.fn("re_std.npy"), arr=np.asarray(window_re_stds))
    np.save(file=job.fn("re_array.npy"), arr=np.asarray(window_re_arrays))
    np.save(file=job.fn("s2_order.npy"), arr=np.asarray(window_order_param))
    np.save(file=job.fn("potential_energy.npy"), arr=np.asarray(potential_energy))

    job.doc.averaged = False

if __name__ == "__main__":
    MyProject().main()
