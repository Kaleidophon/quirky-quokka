"""
Define the hyperparameters for different environments.
"""

HYPERPARAMETERS = {
    "CartPole-v1": {
        "batch_size": 64,
        "discount_factor": 0.8,
        "learn_rate": 1e-3,
        "num_hidden": 256,
        "memory_size": 10000
    },
    "MountainCar-v0": {
        "batch_size": 128,
        "discount_factor": 0.99,
        "learn_rate": 0.001,
        "num_hidden": 128,
        "memory_size": 10000
     },
    "MountainCarContinuous-v0": {
        "batch_size": 128,
        "discount_factor": 0.99,
        "learn_rate": 0.001,
        "num_hidden": 128,
        "memory_size": 10000
     },
    "Acrobot-v1": {
        "batch_size": 128,
        "discount_factor": 0.99,
        "learn_rate": 0.001,
        "num_hidden": 128,
        "memory_size": 10000
    },
    "Pendulum-v0": {
        "batch_size": 128,
        "discount_factor": 0.99,
        "learn_rate": 0.001,
        "num_hidden": 128,
        "memory_size": 10000
    }
}
