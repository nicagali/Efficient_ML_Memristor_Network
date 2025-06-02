G00030001 random graph with three inputs [0, 5, 2] and two desired outputs [3,4].

---> SIMULATION PARAMETERS
training_steps = 200
training_type = 'allostery'
weight_type_vec = ['length', 'radius_base', 'rho', 'pressure']
delta_weight_vec = [1e-3, 1, 1e-4, 1e-3]
learning_rate_vec = [5e-6, 5e-6, 1e-3, 3e2]