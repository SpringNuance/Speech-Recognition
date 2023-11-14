function [X] = gmmb_sample(classes, S)
%GMMB_SAMPLE Sample from a GMM
%   classes = a vector of classes
%   S = GMM trained with GMM Bayes toolkit
%   For each element in classes, samples a vector from S with that class
%   Returns these sampled vectors as rows in X
%
% Author: Aku Rouhe (2018) aku.rouhe@aalto.fi

possible_components = 1:length(S(1).weight);
num_dimensions = length(S(1).mu(:,1));
% First, randomly pick a gaussian to sample from:
chosen_components = arrayfun(...
    @(class) randsample(possible_components, 1, true, S(class).weight), ...
    classes);

% Then sample from that multivariate gaussian
X = zeros(length(classes), num_dimensions);
for i = 1:length(classes)
    class = classes(i);
    component = chosen_components(i);
    X(i,:) = mvnrnd( S(class).mu(:,component),...
        S(class).sigma(:,:,component));
end