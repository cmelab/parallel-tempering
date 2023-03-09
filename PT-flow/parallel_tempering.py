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


class PT_Project(FlowProject):
    pass


class Borah(DefaultSlurmEnvironment):
    hostname_pattern = "borah"
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


def submit_sims(project):
    for job in get_sim_jobs():
        job.doc["done"] = False
    try:
        project.submit()
        print("----------------------")
        print("Successfully submitted simulations...")
        print("----------------------")
    except Exception as error:
        raise RuntimeError(f"project submission failed. Error at line: {error.args[0]}")


@directives(executable="python -u")
@directives(ngpu=0)
@PT_Project.operation
@PT_Project.post(finished)
def sample(job):
    with job:
        print("-----------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("-----------------------")
        # import files from signac flow
        project = signac.get_project()
        path = os.path.join(project.root_directory(), "../{}/".format(job.sp.mode))
        sys.path.append(path)
        from init import init_jobs
        from project import MyProject

        first_wait = job.sp.first_wait
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
                print("----------------------")
                print("Successfully initiated {} project...".format(job.sp.mode))
                print("----------------------")
            except Exception as error:
                raise RuntimeError("project init failed. {}".format(error.args[0]))

            submit_sims(MyProject())

        # load simulation jobs and e_factors
        project = signac.get_project()
        swap_parameters = []
        sim_jobs = []
        for v, s_job in project.find_jobs({"doc.job_type": "sim"}).groupby(job.sp.group_by):
            swap_parameters.append(v)
            sim_jobs.append(list(s_job)[0])
        print('sim_jobs: ', sim_jobs)
        while job.doc["current_attempt"] <= job.sp.n_attempts:
            print('current swap: ', job.doc["current_attempt"])
            # First, making sure the simulations are finished
            print("checking status of simulations...")
            if job.doc["current_attempt"] == 0:
                # set wait time to be longer for the very first run that includes the mixing phase
                wait_time = first_wait
            else:
                wait_time = init_wait
            if check_status(wait_time, extra_wait, max_tries):
                if job.doc["current_attempt"] > 0:
                    # find the last swap
                    last_swap = job.doc["swap_history"][-1]
                    # update last swap information
                    update_swap_info(last_swap)

                # attempting a swap
                print("----------------------")
                print("Initiating a swap...")
                print("----------------------")
                # find a random job and a neighbor to swap their configurations
                i = random.randint(1, len(sim_jobs) - 1)
                # find the neighbor with lower e_factor (equivalent to higher T)
                j = i - 1
                param_i = swap_parameters[i]
                job_i = sim_jobs[i]

                param_j = swap_parameters[j]
                job_j = sim_jobs[j]
                # TODO: get potential energy for both and calculate acceptance criteria.
                print("----------------------")
                print(f"Swapping {job.doc.swap_parameter} {param_i} with {param_j}...")
                print("----------------------")
                # Accepting the swap
                job.doc["swap_history"].append({"i": i, "j": j, "param_i": param_i, "param_j": param_j,
                                                "job_i": job_i.id, "job_j": job_j.id, "done": False})
                job.doc["current_attempt"] += 1

                snapshot_i = gsd.hoomd.open(job_i.fn("restart.gsd"))[0]
                positions_i = copy.deepcopy(snapshot_i.particles.position)

                snapshot_j = gsd.hoomd.open(job_j.fn("restart.gsd"))[0]
                positions_j = copy.deepcopy(snapshot_j.particles.position)

                # swap positions and save snapshot TODO: do we need to swap anything else for MD?
                with gsd.hoomd.open(job_i.fn("restart.gsd"), "wb") as traj:
                    snapshot_i.particles.position = positions_j
                    traj.append(snapshot_i)
                with gsd.hoomd.open(job_j.fn("restart.gsd"), "wb") as traj:
                    snapshot_j.particles.position = positions_i
                    traj.append(snapshot_j)

                # submit simulations
                submit_sims(MyProject())

        job.doc["done"] = True


if __name__ == "__main__":
    PT_Project().main()
