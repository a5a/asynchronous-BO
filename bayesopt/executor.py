"""
This module contains the utility classes for asynchronous Bayesian optimization
"""
import time
from typing import List, Dict, Union, Optional, Callable
from concurrent import futures

import numpy as np


class ExecutorBase:
    """Base interface for interaction with multiple parallel workers

    The simulator and real async interfaces will subclass this class so that
    their interfaces are the same

    Main way to interact with this object is to queue jobs using
    add_job_to_queue(), to wait until the desired number of jobs have completed
    using run_until_n_free() and to get the results via get_completed_jobs().

    Parameters
    ----------
    n_workers : int
        Number of workers allowed

    verbose
        Verbosity

    Attributes
    ----------
    n_workers : int
        Total number of workers

    n_free_workers : int
        Number of workers without allocated jobs

    n_busy_workers : int
        Number of workers currently executing a job

    """

    def __init__(self, n_workers: int, verbose: bool = False):
        self.verbose = verbose
        self.n_workers = n_workers

        self.n_free_workers = n_workers  # type: int
        self.n_busy_workers = 0  # type: int

        # All currently-queued jobs that are not running
        self._queue = []  # type:list

        # These lists contain the jobs that have been removed from the queue
        # but have not been fully processed (e.g. returned to the main process)
        self._running_tasks = []  # type: list
        self._completed_tasks = []  # type: list

    @property
    def age(self) -> float:
        raise NotImplementedError

    @property
    def is_running(self) -> bool:
        all_tasks_todo = len(self._queue) + len(self._running_tasks)
        if all_tasks_todo > 0:
            return True
        else:
            return False

    @property
    def status(self) -> Dict:
        """Get current state (counts) of async workers.

        Returns
        -------
        Dict
            Fields are 'n_free_workers', 'n_busy_workers',
            'n_running_tasks',
                'n_completed_tasks', n_queue, 't'.
        """
        status = {'n_free_workers': self.n_free_workers,
                  'n_busy_workers': self.n_busy_workers,
                  'n_completed_tasks': len(self._completed_tasks),
                  'n_queue': len(self._queue),
                  't': self.age,
                  'is_running': self.is_running}

        if self.verbose:
            print(f"{self.__class__.__name__}.status:\n{status}")
        return status

    def _validate_job(self, job: dict) -> None:
        assert 'x' in job.keys()
        assert 'f' in job.keys()
        assert callable(job['f'])

    def run_until_n_free(self, n_desired_free_workers) -> None:
        """Run the simulator until a desired number of workers are free

        Parameters
        ----------
        n_desired_free_workers: int

        """
        raise NotImplementedError

    def run_until_empty(self) -> None:
        """Run the simulator until all jobs are completed

        """
        raise NotImplementedError

    def add_job_to_queue(self, job: Union[Dict, List]) -> None:
        """Add a job to the queue

        Parameters
        ----------
        job : dict
            Dictionary with a job definition that is passed to a worker.

            Structure:

                {
                    'x': location of sample,
                    'f': function executing the sample,
                }

        """
        if self.verbose:
            print(f"{self.__class__.__name__}.queue_job: queuing job:\n{job}")
        if isinstance(job, list):
            for j in job:
                self._queue.append(j)
        else:
            self._queue.append(job)

        self._update_internal_state()

    def _update_internal_state(self) -> None:
        """
        Main function that takes care of moving jobs to the correct places
        and setting statuses and counts
        """
        raise NotImplementedError

    def get_completed_jobs(self) -> List:
        """Get the completed tasks and clear the internal list.

        Returns
        -------
        list
            List with dicts of the completed tasks
        """

        if self.verbose:
            print(
                f"{self.__class__.__name__}.get_completed_jobs: Getting "
                f"completed "
                f"jobs")

        out = self._completed_tasks
        self._completed_tasks = []
        return out

    def get_array_of_running_jobs(self) -> np.ndarray:
        """Get a numpy array with each busy location in a row

        Returns
        -------
        numpy array of the busy locations stacked vertically
        """
        list_of_jobs = self.get_list_of_running_jobs()  # type:list
        if len(list_of_jobs) > 0:
            x_busy = np.vstack([job['x']
                                for job in list_of_jobs])
        else:
            x_busy = None

        return x_busy

    def get_list_of_running_jobs(self) -> List:
        """Get the currently-running tasks

        Returns
        -------
        List with dicts of the currently-running tasks
        """
        if self.verbose:
            print(f"{self.__class__.__name__}.get_running_jobs")
        return self._running_tasks


class SimExecutor(ExecutorBase):
    """Simulated Async execution of expensive evaluations.

    This class will hold info about the async workers and allow the BO
    algo to interact with them and find out which jobs are currently running

    Methods
    -------
    simulate_next_step()
        Main function to update the simulator. Running this increments the
        simulated time and completes jobs whose duration has completed

    """

    def __init__(self, n_workers: int = 2,
                 time_func: Optional[Callable] = None,
                 verbose=False):
        super().__init__(n_workers=n_workers, verbose=verbose)

        self._time_func = time_func

        # Linearly increasing counter keeping track of "time"
        self._time_ticker = 0

        # This list will hold a counter that will decrement each time the
        # status is checked (simulating waiting for the job to complete)
        self._running_jobs_timings = np.array([])

    @property
    def age(self) -> float:
        return self._time_ticker

    @property
    def status(self) -> Dict:
        """Get current state (counts) of async workers.

        Returns
        -------
        Dict
            Fields are 'n_free_workers', 'n_busy_workers', 'n_running_tasks',
                'n_completed_tasks', n_queue, 't'.
        """
        self._update_internal_counts()
        status = {'n_free_workers': self.n_free_workers,
                  'n_busy_workers': self.n_busy_workers,
                  'n_completed_tasks': len(self._completed_tasks),
                  'n_queue': len(self._queue),
                  't': self.age,
                  'is_running': self.is_running}

        if self.verbose:
            print(f"{self.__class__.__name__}.status:\n{status}")
        return status

    def simulate_next_step(self) -> None:
        """Moves the timeline ahead one tick

        If jobs have finished, then they are inserted into self.completed_jobs.
        If workers are free, they will pick up jobs from the queue
        """
        raise NotImplementedError

    def run_until_n_free(self, n_desired_free_workers) -> None:
        """Run the simulator until a desired number of workers are free

        Parameters
        ----------
        n_desired_free_workers: int

        """
        if self.verbose:
            print(f"{self.__class__.__name__}"
                  f".run_until_free({n_desired_free_workers})")
        while self.n_free_workers < n_desired_free_workers:
            self.simulate_next_step()

    def run_until_empty(self):
        """Run the simulator until all jobs are completed

        """
        if self.verbose:
            print(f"{self.__class__.__name__}.run_until_empty()")
        while self.n_free_workers < self.n_workers:
            self.simulate_next_step()

    def add_job_to_queue(self, job: Union[Dict, List], update=True) -> None:
        """Add a job to the queue

        Parameters
        ----------
        job : dict
            Dictionary with a job definition that is passed to a worker.

            Structure:

                {
                    'x': location of sample,
                    'f': function executing the sample,
                    OPTIONAL:
                    't': how long this task will take
                }

        update
            whether or not to run the internal update func. This is useful
            in case a list of jobs is provided, as then this function
            is called recursively and shouldn't update until all jobs
            have been added

        """
        if self.verbose:
            print(f"{self.__class__.__name__}.queue_job: queuing job:\n{job}")

        if isinstance(job, list):
            for j in job:
                self.add_job_to_queue(j, update=False)
        else:
            if 't' not in job.keys():
                job = self._add_running_time_to_job(job)

            self._queue.append(job)

        if update:
            self._update_internal_state()

    def _update_internal_state(self) -> None:
        """
        Main function that takes care of moving jobs to the correct places
        and setting statuses and counts
        """
        # Useful here if this is the first call to the async worker and
        # nothing is running
        self._begin_queued_job_if_workers_free()

        # If jobs have completed, update the internal lists and counters
        completed_tasks_since_last_check = \
            np.where(self._running_jobs_timings <= 0)[0]

        if self.verbose:
            print(
                f"{self.__class__.__name__}.update_internal_state: Checking "
                f"for completed tasks... found "
                f"{len(completed_tasks_since_last_check)}")
            print(
                f"{self.__class__.__name__}.update_internal_state: Current "
                f"jobs time:\n{self._running_jobs_timings}")

        if len(completed_tasks_since_last_check) > 0:
            if self.verbose:
                print(
                    f"{self.__class__.__name__}.update_internal_state: "
                    f"Dealing with completed tasks...")
            # remove from running tasks, starting from the end of the array
            for idx in np.sort(completed_tasks_since_last_check)[::-1]:
                new_completed_task = self._get_result(
                    self._running_tasks[idx])
                self._completed_tasks.append(new_completed_task)
                self._running_tasks.pop(idx)
                self._running_jobs_timings = \
                    np.delete(self._running_jobs_timings, idx)

            self._update_internal_counts()
        self._begin_queued_job_if_workers_free()
        if self.verbose:
            print(f"{self.__class__.__name__}.update_internal_state: Updated "
                  f"running times:\n{self._running_jobs_timings}")

    def _begin_queued_job_if_workers_free(self) -> None:
        # if any workers are free, pop next job from top of queue and
        # insert into running tasks
        while self.n_free_workers > 0 and len(self._queue) > 0:
            if self.verbose:
                print(
                    f"{self.__class__.__name__}."
                    f"_begin_queued_job_if_workers_free: "
                    f"Moving task from queue to running.")
            self._begin_task(self._queue.pop(0))

    def _update_internal_counts(self) -> None:
        # TODO: test this
        self.n_busy_workers = len(self._running_tasks)
        self.n_free_workers = self.n_workers - self.n_busy_workers

    def _begin_task(self, task) -> None:
        """Begins the given task

        Assigns the task to one of the free workers. Updates the internal
        counters and lists

        Parameters
        ----------
        task : dict
            Task with structure as defined in self.queue_job()

        """

        if self.verbose:
            print(
                f"{self.__class__.__name__}._begin_task: adding task to "
                f"running jobs: "
                f"\n{task}")
        # TODO: test this
        self._running_tasks.append(task)

        t = task['t']
        self._running_jobs_timings = np.hstack((self._running_jobs_timings, t))

        self._update_internal_counts()

    def _get_result(self, task) -> dict:

        if self.verbose:
            print(f"{self.__class__.__name__}.execute_task: executing task "
                  f"\n{task}")
        # TODO: test this
        task['y'] = task['f'](np.atleast_2d(task['x']))
        return task

    def _add_running_time_to_job(self, job) -> dict:
        if self._time_func is not None:
            job['t'] = self._time_func()
        else:
            job['t'] = np.random.rand()
        return job


class SimExecutorIntegerTicks(SimExecutor):
    """Async simulator with integer time incrementing
    """

    def __init__(self, n_workers: int = 2, time_func=None, verbose=False):
        super().__init__(n_workers=n_workers,
                         time_func=time_func,
                         verbose=verbose)

    def simulate_next_step(self) -> None:
        """Moves the timeline ahead one tick

        If jobs have finished, then they are inserted into self.completed_jobs.
        If workers are free, they will pick up jobs from the queue
        """
        # Time moves ahead one tick, so update each job's time left
        self._time_ticker += 1
        self._running_jobs_timings -= 1
        self._update_internal_state()


class SimExecutorJumpToCompletedJob(SimExecutor):
    """Async simulator that jumps to the next completed job each time it
    is "incremented"
    """

    def __init__(self, n_workers: int = 2,
                 time_func: Optional[Callable] = None,
                 verbose=False):
        super().__init__(n_workers=n_workers,
                         time_func=time_func,
                         verbose=verbose)

    def simulate_next_step(self) -> None:
        """Moves the timeline ahead so that the next job is completed

        If jobs have finished, then they are inserted into self.completed_jobs.
        If workers are free, they will pick up jobs from the queue
        """
        if self.verbose:
            print(f"{self.__class__.__name__}.simulate_next_step()")

        # Time moves ahead one tick, so update each job's time left
        increment = np.min([t for t in self._running_jobs_timings if t > 0])
        self._time_ticker += increment
        self._running_jobs_timings -= increment
        self._update_internal_state()


class JobExecutor(ExecutorBase):
    """Async controller that interacts with external async function calls

    Will be used to run ML algorithms in parallel for synch and async BO

    Functions that run must take in a job dict and return the same
    job dict with the result ['y'] and runtime ['t'].
    """

    def __init__(self, n_workers: int, polling_frequency=0.5, verbose=False):
        super().__init__(n_workers, verbose=verbose)

        self._creation_time = time.time()
        self._polling_delay = polling_frequency
        self._executor = futures.ProcessPoolExecutor(n_workers)
        self._futures = []

    @property
    def age(self) -> float:
        return time.time() - self._creation_time

    def run_until_n_free(self, n_desired_free_workers) -> None:
        """Wait until a desired number of workers are free

        Parameters
        ----------
        n_desired_free_workers: int

        """
        if self.verbose:
            print(f"{self.__class__.__name__}"
                  f".run_until_free({n_desired_free_workers})")
        while self.n_free_workers < n_desired_free_workers:
            time.sleep(self._polling_delay)
            self._update_internal_state()

    def run_until_empty(self) -> None:
        """Run the simulator until all jobs are completed

        """

        if self.verbose:
            print(f"{self.__class__.__name__}"
                  f".run_until_empty()")
        while self.n_free_workers < self.n_workers:
            time.sleep(self._polling_delay)
            self._update_internal_state()

    def _update_internal_state(self) -> None:
        """
        Setting internal counts
        """
        self._clean_up_completed_processes()
        self._begin_jobs_if_workers_free()

        # Update internal counts
        self.n_free_workers = self.n_workers - len(self._running_tasks)
        self.n_busy_workers = len(self._running_tasks)

    def _clean_up_completed_processes(self) -> None:
        """
        Remove completed jobs from the current processes and save results
        """
        if len(self._futures) > 0:
            idx_complete = np.where([not f.running()
                                     for f in self._futures])[0]
            # Going through the idx in reverse order, as we're using pop()
            for ii in np.sort(idx_complete)[::-1]:
                f_complete = self._futures.pop(ii)  # type: futures.Future
                complete_job_dict = self._running_tasks.pop(ii)
                complete_job_dict['y'] = f_complete.result()
                self._completed_tasks.append(complete_job_dict)

    def _begin_jobs_if_workers_free(self) -> None:
        """
        If workers are free, start a job from the queue
        """
        # If workers are free and queue is not empty, start jobs
        while len(self._futures) < self.n_workers and len(self._queue) > 0:
            self._futures.append(self._submit_job_to_executor(0))

    def _submit_job_to_executor(self, index) -> futures.Future:
        """Submits the chosen job from the queue to the executor

        Parameters
        ----------
        index
            Index in the queue of the job to be executed

        Returns
        -------
        Future object of the submitted job
        """
        job = self._queue.pop(index)  # type: dict
        self._validate_job(job)
        self._running_tasks.append(job)
        future = self._executor.submit(job['f'], job['x'])
        return future


class JobExecutorInSeries(JobExecutor):
    """Interface that runs the jobs in series
    but acts like a batch-running interface to the outside.

    self._futures is not a list of futures any more. This is a placeholder
    for the jobs that have yet to run to complete the batch
    """

    def __init__(self, n_workers: int, polling_frequency=0.5, verbose=False):
        super().__init__(n_workers, polling_frequency=polling_frequency,
                         verbose=verbose)

        # Overwrite the executor to actually only run one job at a time
        self._executor = futures.ProcessPoolExecutor(1)
        # self._batch = []

    def _clean_up_completed_processes(self) -> None:
        """
        Remove completed jobs from the current processes and save results
        """
        if len(self._futures) > 0:
            is_complete = self._futures[0].running()
            if is_complete:
                f_complete = self._futures.pop(0)  # type: futures.Future
                complete_job_dict = self._running_tasks.pop(0)
                complete_job_dict['y'] = f_complete.result()
                self._completed_tasks.append(complete_job_dict)

    def _begin_jobs_if_workers_free(self) -> None:
        """
        If workers are free, start a job from the queue
        """
        # If worker is free and queue is not empty, start job
        if len(self._futures) == 0:
            if len(self._running_tasks) > 0:
                job = self._running_tasks[0]
                self._validate_job(job)
                self._futures.append(
                    self._executor.submit(job['f'], job['x'])
                )
            else:
                while len(self._queue) > 0 and \
                        len(self._running_tasks) < self.n_workers:
                    self._running_tasks.append(self._queue.pop(0))


class JobExecutorInSeriesBlocking(ExecutorBase):
    """Interface that runs the jobs in series and blocks execution of code
    until it's done
    """

    def __init__(self, n_workers: int, verbose=False):
        super().__init__(n_workers, verbose=verbose)
        self._creation_time = time.time()

    def run_until_n_free(self, n_desired_free_workers) -> None:
        """Run the simulator until a desired number of workers are free

        Parameters
        ----------
        n_desired_free_workers: int

        """
        while self.n_free_workers < n_desired_free_workers:
            self.run_next()

    def run_until_empty(self) -> None:
        """Run the simulator until all jobs are completed

        """
        while self.n_free_workers < self.n_workers:
            self.run_next()

    def _update_internal_state(self):

        while len(self._running_tasks) < self.n_workers and len(
                self._queue) > 0:
            self._running_tasks.append(self._queue.pop(0))

        self.n_busy_workers = len(self._running_tasks)
        self.n_free_workers = self.n_workers - self.n_busy_workers

    def run_next(self):
        self._move_tasks_from_queue_to_running()

        if len(self._running_tasks) > 0:
            job = self._running_tasks.pop(0)
            self._validate_job(job)
            result = job['f'](job['x'])
            job['y'] = result
            self._completed_tasks.append(job)
        self._update_internal_state()

    @property
    def age(self):
        return time.time() - self._creation_time

    def _move_tasks_from_queue_to_running(self):
        # fill up the running tasks from the queue
        while len(self._running_tasks) < self.n_workers and \
                len(self._queue) > 0:
            self._running_tasks.append(self._queue.pop(0))


class AsyncJobExecutorInSeriesBlocking(JobExecutorInSeriesBlocking):
    """Interface that runs the jobs in series and blocks execution of code
    until it's done

    To the "outside" it looks like jobs are running asynchronously.
    The Executor's time variable is updated as if jobs were running in async.
    """

    def __init__(self, n_workers: int, verbose=False):
        super().__init__(n_workers, verbose=verbose)

        # The job runtimes will be the same size as self._running_tasks. As
        # a job is running, its runtime will be np.nan. The executor will
        # continue running jobs until all of its entries are numeric values
        # (not NaN). Then it will pop the jobs as required. As soon as a new
        # job is added to self._running_tasks, it is run. The Async
        # behaviour is enforced using this variable.
        # When a job is removed from this list, all remaining time values
        # are reduced by its amount.
        self._job_runtimes = []

        # This variable will hold our effective async time
        self._time = 0

    @property
    def age(self):
        return self._time

    def _update_internal_state(self):
        self._move_tasks_from_queue_to_running()

        self.n_busy_workers = len(self._running_tasks)
        self.n_free_workers = self.n_workers - self.n_busy_workers

    def run_next(self):
        self._move_tasks_from_queue_to_running()

        # If there are jobs that haven't run yet
        if np.sum(np.isnan(self._job_runtimes)) > 0:
            nan_idx = np.where(np.isnan(self._job_runtimes))[0]
            for idx in nan_idx:
                job = self._running_tasks[idx]  # type: dict
                self._validate_job(job)
                t0 = time.time()
                result = job['f'](job['x'])
                time_taken = time.time() - t0
                job['y'] = result
                job['t'] = time_taken
                self._job_runtimes[idx] = time_taken

        # Do the async fudging
        if len(self._running_tasks) > 0:
            next_idx = np.argmin(self._job_runtimes)  # type: int
            job = self._running_tasks.pop(next_idx)
            job_time = self._job_runtimes.pop(next_idx)
            self._time += job_time
            self._job_runtimes = [t - job_time for t in self._job_runtimes]
            self._completed_tasks.append(job)

        self._update_internal_state()

    def _move_tasks_from_queue_to_running(self):
        # fill up the running tasks from the queue
        while len(self._running_tasks) < self.n_workers and \
                len(self._queue) > 0:
            self._running_tasks.append(self._queue.pop(0))
            self._job_runtimes.append(np.nan)
