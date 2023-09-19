Evoman is a video game playing framework to be used as a testbed for optimization algorithms.

A demo can be found here:  https://www.youtube.com/watch?v=ZqaMjd1E4ZI


# Making modifications using DEAP (GA)

## Basics

In `test_deap.py` is a minimal example hooked up with the engine. The `main()` function is the entry point of execution of the algorithm, as is, it will run a set number of generatios and then at the end show you the execution in normal time of the best
individual accross generation. It will also print statistics each generation.



## Adding a new method (Selection, Mutation, Crossover)

### Crossover and Mutation
Currently the setup takes into account both off these methods (both will have a chance to be applied),
for now the step size is constants but that can be modified by looking at NEAT documentation. To change 
either method the `toolbox` class needs to be used:

1.  Declare your new mutation function. ex.
```
    def mutate_ex(ind, step):
        for v, i in enumerate(ind):
            ind[i] = v + v*step
        return ind
```
2. Register this function in the `toolbox` under the name *mutate* and give it a parameter value to your static parameters:
```
toolbox.register("mutate", mutate_ex, step=0.5)
```

3. (Optional) Change the probability of mutation (**MUTPB**) or the probability of crossover (**CXPB**) in the code

4. Execute `python3 test_deap.py`

5. Make sure your function is under `deap_algorithms.py` and if the implenmentation required changes to `test_deap.py` make a copy and put your solution there to avoid merge conflicts

### Selection / Replacement

Selection and replacement aren't mutually exclusive but we can choose to only use selection in the replacement strategy. As it is right now selection is a part of the offspring selection but it could also be implenmented in parent selection (as part of mutation ?)
