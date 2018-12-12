# Environment-related difference of Deep Q-Learning and Deep Double Q-Learning

Parameters we used for training in the gym environments `CartPole-v1`, `MountainCar-v0`, `Acrobot-v1`, `Pendulum-v0`:

```
CartPole-v1: 
batch_size: 128,
discount_factor: 0.8,
learn_rate: 1e-3,
num_hidden: 256,
memory_size: 10000,
update_target_q: 10,
max_steps: 200
```


```
MountainCar-v0: 
batch_size: 128,
discount_factor: 0.99,
learn_rate: 0.001,
num_hidden: 128,
memory_size: 10000,
update_target_q: 100,
max_steps: 1000
```


```
Acrobot-v1: 
batch_size: 128,
discount_factor: 0.99,
learn_rate: 0.001,
num_hidden: 128,
memory_size: 10000,
update_target_q: 10,
max_steps: 1000
```

```Pendulum-v0:
batch_size: 128,
discount_factor: 0.9,
learn_rate: 0.001,
num_hidden: 128,
memory_size: 10000,
update_target_q: 10,
max_steps: 2000

```


