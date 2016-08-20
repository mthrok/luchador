Luchador is a agent library for Reinforcement Learning.

It has a bsic implementation of DQN in Theano and Tensorflow.

Training with both Theano or Tensorflow can be monitored via Tensorboard*.

To install luchador, run the follwoing command.

```
git clone http://github.com/mthrok/luchador
cd luchador && pip install .
```

For an example usage, please refer to [simple_dqn luchador version](https://github.com/mthrok/simple_dqn/blob/luchador/src/deepqnetwork_luchador.py).

* To monitor Theano GPU training, Tensorflow must be **GPU DISABLED**. Therefore it is recommended to separate environments with virtual environment tools.
