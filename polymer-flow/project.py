"""Define the project's workflow logic and operation functions.
Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:
    $ python src/project.py --help
"""
import os
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
    hostname_pattern = "fry"
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


@directives(executable="python -u")
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
        restart = job.isfile("restart.gsd")
        molecule_obj = getattr(hoomd_polymers.molecules, job.sp.molecule)
        ff_obj = getattr(hoomd_polymers.forcefields, job.sp.forcefield)
        system_obj = getattr(hoomd_polymers.systems, job.sp.system)
        system = system_obj(
                molecule=molecule_obj,
                n_mols=job.sp.n_chains,
                chain_lengths=job.sp.chain_lengths,
                density=job.sp.density,
                mol_kwargs=job.sp.mol_kwargs
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
        # Set up stuff to initialize a Simulation
        job.doc.gsd_write_frequency = int(
                job.sp.num_gsd_frames / job.doc.total_steps
        ) 
        job.doc.log_write_frequency = int(
                job.sp.num_data_logs / job.doc.total_steps
        ) 
        sim = Simulation(
                initial_state=system.hoomd_snapshot,
                forcefield=system.hoomd_forcefield,
                gsd_write_freq=job.doc.gsd_write_frequency,
                log_write_freq=job.doc.log_write_frequency,
        )
        sim.pickle_forcefield()
        print("----------------------")
        print("Starting simulation...")
        print("----------------------")
        job.doc.target_box_reduced = (
                system.target_box*10/system.reference_values.distance
        )
        # Run shrink simulation
        sim.run_update_volume(
                final_box=job.doc.traget_box_reduced,
                n_steps=job.sp.shrink_steps,
                period=int(job.sp.shrink_period),
                kT=job.sp.shrink_kT,
                tau_kt=job.sp.tau_kt
        )
        assert sim.box_lengths_reduced == job.doc.target_box_reduced
        job.doc.ran_shrink = True
        sim.pickle_state(file_path="shrink_finished.pickle")
        print("----------------------")
        print("Shrink simulation finished...")
        print("----------------------")
        # Run NVT simulation
        sim.run_NVT(
                n_steps=job.sp.n_steps, kT=job.sp.kT, tau_kt=job.sp.tau_kt
        )
        sim.pickle_state(file_path="restart.pickle")
        print("----------------------")
        print("Simulation finished...")
        print("----------------------")
        if job.doc["swap"]:
            copy_trajectory(job, fname=f"trajectory_{job.doc.current_run}_swap.gsd")
        else:
            copy_trajectory(job, fname=f"trajectory_{job.doc.current_run}.gsd")
        job.doc["timestep"].append(sim.timestep)
        job.doc["current_run"] += 1
        sim.reset_system() # What exactly is this func doing, how to implement with polymer sims
        job.doc["done"] = True


if __name__ == "__main__":
    MyProject().main()
