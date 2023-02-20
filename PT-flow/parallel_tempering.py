"""Define the project's workflow logic and operation functions.
Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:
    $ python src/project.py --help
"""
import os
import sys

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
import signac
import random
import copy
import gsd.hoomd
import logging

logger = logging.getLogger()

class MyProject(FlowProject):
    pass


class Borah(DefaultSlurmEnvironment):
    hostname_pattern = "borah"
    template = "../templates/borah.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="short",
            help="Specify the partition to submit to."
        )


class R2(DefaultSlurmEnvironment):
    hostname_pattern = "r2"
    template = "../templates/r2.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="shortq",
            help="Specify the partition to submit to."
        )


class Fry(DefaultSlurmEnvironment):
    hostname_pattern = "fry"
    template = "../templates/fry.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="batch",
            help="Specify the partition to submit to."
        )


# Definition of project-related labels (classification)
@MyProject.label
def finished(job):
    return job.doc.get("done")


@MyProject.label
def initialized(job):
    return job.doc.get("current_attempt") > 0


@directives(executable="python -u")
@MyProject.operation
@MyProject.post(finished)
def sample(job):
    with job:
        print("-----------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("-----------------------")
        # import files from signac flow
        sys.path.append("../{}/".format(job.sp.mode))
        from init import init_project
        from project import MyProject

        # Before first swap attempt, first we need to initiate signac project and submit jobs.
        if job.doc["current_attempt"] == 0:
            print("----------------------")
            print("Starting Parallel tempering (First run)...")
            print("----------------------")
            print("----------------------")
            print("Initiating {} project...".format(job.sp.mode))
            print("----------------------")

            try:
                init_project()
                print("----------------------")
                print("Successfully initiated {} project...".format(job.sp.mode))
                print("----------------------")
            except Exception as error:
                logger.error(f"Error at line: {error.args[0]}")
                raise RuntimeError("project init failed")

            try:
                MyProject().submit()
                print("----------------------")
                print("Successfully submitted {} project...".format(job.sp.mode))
                print("----------------------")
                logger.info("Successfully submitted {} project...".format(job.sp.mode))
            except Exception as error:
                logger.error(f"Error at line: {error.args[0]}")
                raise RuntimeError("project submission failed")
            #TODO: If we can find a way to check the status of submitted jobs frequently here
            # (maybe using a while loop), then we can move on to the next step easily.

        else:
            # find signac project
            project = signac.get_project(job.sp.workspace)
            jobs_list = project.find_jobs().groupby("e_factor")

            # First, making sure the previous swap (if exists) was finished successfully
            if job.doc["current_attempt"] > 0:
                # find the last swap
                last_swap = job.doc["swap_history"][-1]

                job_i = project.open_job(id=last_swap["job_i"])
                job_j = project.open_job(id=last_swap["job_j"])

                # TODO: probably need to check all the jobs, but for now we only check i & j status
                try:
                    assert job_i.doc["done"] and job_j.doc["done"]
                    job.doc["swap_history"][-1]["done"] = True
                except AssertionError:
                    # previous swap was unsuccessful, need to repeat.
                    job.doc["current_attempt"] -= 1

            if job.doc["current_attempt"] <= job.sp.n_attempts:
                print("----------------------")
                print("Initiating a swap...")
                print("----------------------")
                # find a random job and a neighbor to swap their configurations
                i = random.randint(1, len(jobs_list) - 1)
                # find the neighbor with lower e_factor (equivalent to higher T)
                j = i - 1
                e_factor_i = jobs_list[i][0]
                job_i = list(jobs_list[i][1])[0]

                e_factor_j = jobs_list[j][0]
                job_j = list(jobs_list[j][1])[0]
                # TODO: get potential energy for both and calculate acceptance criteria.

                print("----------------------")
                print("Swapping e_factor {} with {}...".format(e_factor_i, e_factor_j))
                print("----------------------")
                # Accepting the swap
                job.doc["swap_history"].append({"i": i, "j": j, "e_factor_i": e_factor_i, "e_factor_j": e_factor_j,
                                                "job_i": job_i, "job_j": job_j, "done": False})
                job.doc["current_attempt"] += 1

                snapshot_i = gsd.hoomd.open(job_i.fn("snapshot.gsd"))[0]
                positions_i = copy.deepcopy(snapshot_i.particles.position)

                snapshot_j = job_j.fn("snapshot.gsd")[0]
                positions_j = copy.deepcopy(snapshot_i.particles.position)

                # swap positions and save snapshot
                with gsd.hoomd.open(job_i.fn("snapshot.gsd"), "w") as traj:
                    snapshot_i.particle.position = positions_j
                    traj.append(snapshot_i)

                with gsd.hoomd.open(job_j.fn("snapshot.gsd"), "w") as traj:
                    snapshot_j.particle.position = positions_i
                    traj.append(snapshot_j)

                # change state of all jobs
                for e_factor, sample_job in jobs_list:
                    sample_job = list(sample_job)[0]
                    sample_job.sp.n_steps = job.sp.PT_n_steps
                    sample_job.sp.kT = job.sp.PT_kT
                    sample_job.doc["done"] = False
                # resume signac jobs
                try:
                    MyProject().submit()
                    print("----------------------")
                    print("Successfully resumed {} project...".format(job.sp.mode))
                    print("----------------------")
                    logger.info("Successfully resumed {} project...".format(job.sp.mode))
                except Exception as error:
                    logger.error(f"Error at line: {error.args[0]}")
                    raise RuntimeError("project resume failed")




            else:
                job.doc["done"] = True


if __name__ == "__main__":
    MyProject().main()
