"""Define the project's workflow logic and operation functions.
Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:
    $ python src/project.py --help
"""
import os
import shutil
import sys
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
def initialized(job):
    return job.isfile("trajectory.gsd")


@MyProject.label
def is_sim(job):
    return job.doc["job_type"] == "sim"


# Useful functions
def copy_trajectory(job, fname):
    shutil.copyfile(job.fn("trajectory.gsd"), job.fn(f"{fname}"))


def load_pickle_ff(job, ff_file):
    """
    Load hoomd snapshot and list of hoomd forces
    from pickle files in a job's workspace.
    """
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
            hoomd_ff = load_pickle_ff(job, "forcefield.picke")
        else: # No restart, generate the system and apply a FF
            molecule_obj = getattr(hoomd_polymers.molecules, job.sp.molecule)
            ff_obj = getattr(hoomd_polymers.forcefields, job.sp.forcefield)
            system_obj = getattr(hoomd_polymers.systems, job.sp.system)
            system = system_obj(
                    molecule=molecule_obj,
                    n_mols=job.sp.n_chains,
                    chain_lengths=job.sp.chain_lengths,
                    density=job.sp.density,
                    mol_kwargs=job.sp.molecule_kwargs,
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
                    make_charge_neutral=True
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
        if not job.fn("forcefield.pickle"): # Pickle FF for future runs
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
                    system.target_box*10/system.reference_values.distance
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
            copy_trajectory(job, fname=f"trajectory_{job.doc.current_run}_swap.gsd")
        else:
            copy_trajectory(job, fname=f"trajectory_{job.doc.current_run}.gsd")
        job.doc["timestep"].append(sim.timestep)

        job.doc["current_run"] += 1
        job.doc["done"] = True


if __name__ == "__main__":
    MyProject().main()
