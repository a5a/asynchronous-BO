# asynchronous-BO
Code related to the paper "Asynchronous Batch Bayesian Optimisation with Improved Local Penalisation" by Ahsan S. Alvi, Binxin Ru, Jan Calliess, Stephen J. Roberts, Michael A. Osborne appearing in the proceedings of ICML 2019. 

Paper link: https://arxiv.org/abs/1901.10452


### Structure
- bayesopt
    - acquisition.py: Acquisition function classes, e.g. expected improvement, local penalisation, etc.
    - async_bo.py: Base class for asynchronous BO, as well as all the different permutations, e.g. PLAyBOOK, TS, etc.
    - batch_bo.py: Subclasses of classes in async_bo. Synchronous batch BO.
    - bayesopt.py: Base class for BO. Also acts as the standard sequential BO class
    - executor.py: classes that perform function evaluations in parallel (synch or async is defined in the BO class)
    - exp_utils.py: utilities e.g. creating BO classes quickly, getting the correct task and synth time func
    - util.py: MES-related functions, hallucination functions
- exps
    - Interface for async and sync BO exps
       

### How to use this package
- BO experiments are run by executing exps/exp_async_synch_math_func.py with desired parameters
         

### Dependencies
- numpy
- scipy
- pandas
- matplotlib
- scipydirect
- tqdm
- GPy
- ml_utils
