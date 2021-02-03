## QUBO - NN

QUBO matrices are used to describe an optimization problem as a matrix such that a Quantum Annealer (such as a D-Wave QA) can solve it.

Now, these matrices are quite an interesting construct.. Thus, a few questions arise:

* Is it possible to classify the problem class based on the QUBO matrix?
* What is the exact trade-off when auto-encoding QUBO matrices, i.e. who far can one go before the solution quality drops significantly?

Let's find out.

## Project Structure

|File|Purpose|
|----|-------|
|nn/|Contains neural network models|
|problems/|Contains generators and evaluators for specific problems such as 3SAT or TSP|
|qubo/|Will contain the generic QUBO library|
|config.py|Configuration (json) handling|
|main.py|Main entry point|
|pipeline.py|End to end training and testing of NNs on QUBO matrices|
|simulations.json|All experiments and configurations|

Problems implemented so far:

* Number Partitioning
* Maximum Cut

## Contributing

Pull requests are very welcome. Before submitting one, run all tests with `python3 -m unittest` and make sure nothing is broken.

## Using

TODO

```
usage: main.py [-h] type cmd cfg_id

python3 -m qubo_nn.main classify train 1
```

## References

```
Glover, Fred, Gary Kochenberger, and Yu Du. "A tutorial on formulating and using qubo models." arXiv preprint arXiv:1811.11538 (2018).
```

## Related Work

[Hadamard Gate Transformation for 3 or more QuBits](https://blog.xa0.de/post/Hadamard-Gate-Transformation-for-3-or-more-QuBits/)
