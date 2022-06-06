############### Configuration file for Bayesian ###############
layer_type = 'bbb'  # 'bbb' or 'lrt'
activation_type = 'relu'  # 'softplus' or 'relu'

priors={
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}

arch = 'resnet50'

n_epochs = 200
lr_start = 0.001
num_workers = 4
valid_size = 0.2
batch_size = 256
train_ens = 5
valid_ens = 100
beta_type = 0.1  # 'Blundell', 'Standard', etc. Use float for const value
