%TRAIN_GMM
%
%    S = TRAIN_GMM(data, class, num_comp)
%
%    Trains a Gaussian mixture model using the GMM Bayes Toolbox.
%    Returns the structure containing the model parameters. Really
%    just a wrapper to gmmb_create function.
%
%      data      - Training samples
%      class     - Class numbers of the training samples
%      num_comp  - Number of Gaussian components in each mixture

function S = train_gmm(data, class, num_comp)

  S = gmmb_create(data, class, 'EMD', 'components', num_comp, 'maxloops', 25, 'init', 'cmeans1');
