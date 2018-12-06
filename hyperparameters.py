"""
Define the hyperparameters for different environments.
"""

HYPERPARAMETERS = {
    "CartPole-v1": {
        "batch_size": 64,
        "discount_factor": 0.8,
        "learn_rate": 1e-3,
        "num_hidden": 256,
        "memory_size": 10000,
        "update_target_q": 10
    }
}
