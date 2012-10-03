function mu = assignmentToMarginal(x, num_states)

mu = zeros(num_states, numel(x));
idx = sub2ind(size(mu), x(:)', [1:numel(x)]);
mu(idx) = 1;

% TODO: cell version of this!
