"""Define the project's workflow logic and operation functions.
Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:
    $ python src/project.py --help
"""
import os
import sys
import time

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


class PT_Project(FlowProject):
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
@PT_Project.label
def finished(job):
    return job.doc.get("done")


@PT_Project.label
def initialized(job):
    return job.doc.get("current_attempt") > 0


def wait(init_wait, extra_wait, max_tries):
    """
    Waits for a certain amount of time for the simulations to finish.
    :param init_wait: Initial wait time before first status check.
    :param extra_wait: Wait time between status checks.
    :param max_tries: Maximum number of status checks.
    :return:
    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            retries = 0
            start_init = time.time()
            print('initial sleep.....')
            time.sleep(init_wait)
            print('init wait time: ', time.time() - start_init)
            while retries < max_tries:
                print('try: ', retries)
                value = function(*args, **kwargs)
                if value:
                    return value
                else:
                    start_add = time.time()
                    print('additional sleep.....')
                    time.sleep(extra_wait)
                    print('additional wait: ', time.time() - start_add)
                    retries += 1
            return False

        return wrapper

    return decorator


def get_sim_jobs():
    project = signac.get_project()
    jobs_list = project.find_jobs({'doc.job_type': "sim"})
    return jobs_list


def check_status(init_wait, extra_wait, max_tries):
    @wait(init_wait, extra_wait, max_tries)
    def _check_status():
        return all([job.doc["done"] for job in get_sim_jobs()])

    return _check_status()


def update_swap_info(swap):
    swap["done"] = True
    project = signac.get_project()
    job_i = project.open_job(id=swap["job_i"])
    job_j = project.open_job(id=swap["job_j"])
    # job i and j from previous swap no longer need the swap flag to be True.
    job_i.doc["swap"] = False
    job_j.doc["swap"] = False


def submit_sims(job, project):
    for job in get_sim_jobs():
        job.doc["done"] = False
    try:
        project.submit()
        print("----------------------")
        print("Successfully submitted {} project...".format(job.sp.mode))
        print("----------------------")
        logger.info("Successfully submitted {} project...".format(job.sp.mode))
    except Exception as error:
        logger.error(f"Error at line: {error.args[0]}")
        raise RuntimeError("project submission failed")


@directives(executable="python -u")
@PT_Project.operation
@PT_Project.post(finished)
def sample(job):
    with job:
        print("-----------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("-----------------------")
        # import files from signac flow
        sys.path.append("../{}/".format(job.sp.mode))
        from init import init_jobs
        from project import MyProject

        init_wait = job.sp.init_wait
        extra_wait = job.sp.extra_wait
        max_tries = job.sp.max_tries
        # Before first swap attempt, first we need to initiate signac project and submit jobs.
        if job.doc["current_attempt"] == 0:
            print("----------------------")
            print("Starting Parallel tempering (First run)...")
            print("----------------------")
            print("----------------------")
            print("Initiating {} project...".format(job.sp.mode))
            print("----------------------")

            try:
                init_jobs()
                logger.info("Successfully initiated {} project...".format(job.sp.mode))
                print("----------------------")
                print("Successfully initiated {} project...".format(job.sp.mode))
                print("----------------------")
            except Exception as error:
                logger.error(f"Error at line: {error.args[0]}")
                raise RuntimeError("project init failed. {}".format(error.args[0]))

            submit_sims(job, MyProject())

        while job.doc["current_attempt"] <= job.sp.n_attempts:
            print('current swap: ', job.doc["current_attempt"])
            # First, making sure the simulations are finished
            print("checking status of simulations...")
            if check_status(init_wait, extra_wait, max_tries):
                if job.doc["current_attempt"] > 0:
                    # find the last swap
                    last_swap = job.doc["swap_history"][-1]
                    # update last swap information
                    update_swap_info(last_swap)

                # attempting a swap
                print("----------------------")
                print("Initiating a swap...")
                print("----------------------")
                project = signac.get_project()
                sim_jobs = project.find_jobs({"doc.job_type": "sim"}).groupby("e_factor")
                # find a random job and a neighbor to swap their configurations
                i = random.randint(1, len(sim_jobs) - 1)
                # find the neighbor with lower e_factor (equivalent to higher T)
                j = i - 1
                e_factor_i = sim_jobs[i][0]
                job_i = list(sim_jobs[i][1])[0]

                e_factor_j = sim_jobs[j][0]
                job_j = list(sim_jobs[j][1])[0]
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

                # submit simulations
                submit_sims(job, MyProject())

        job.doc["done"] = True


if __name__ == "__main__":
    PT_Project().main()
