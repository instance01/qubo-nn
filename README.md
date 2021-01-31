## QUBO - NN

QUBO matrices are used to describe an optimization problem as a matrix such that a Quantum Annealer (such as a D-Wave QA) can solve it.

Now, these matrices are quite an interesting construct.. Thus, a few questions arise:

* Is it possible to classify the problem class based on the QUBO matrix?
* What is the exact trade-off when auto-encoding QUBO matrices, i.e. who far can one go before the solution quality drops significantly?

Let's find out.

## Project structure

|File|Purpose|
|----|-------|
|qubo\_nn.py|Main entry point|
|pipeline.py|End to end training and testing of NNs on QUBO matrices|
|qubo/|Will contain the generic QUBO library|
|nn/|Will contain neural network models|
|problems/|Will contain generators and evaluators for specific problems such as 3SAT or TSP|

## Contributing

Pull requests are very welcome. Before submitting one, run all tests with `python3 -m unittest` and make sure nothing is broken.

## Using

TODO

## References

```
Glover, Fred, Gary Kochenberger, and Yu Du. "A tutorial on formulating and using qubo models." arXiv preprint arXiv:1811.11538 (2018).
```
