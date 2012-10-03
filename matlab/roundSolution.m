function mu = roundSolution(mu)

if (~iscell(mu))
    [dummy, x] = max(mu, [], 1);
    mu = zeros(size(mu));
    ind = sub2ind(size(mu), x(:),[1:size(mu,2)]');
    mu(ind) = 1;
else
    for i=1:numel(mu)
        [dummy, x] = max(mu{i});
        mu{i} = zeros(size(mu{i}));
        mu{i}(x) = 1;
    end
end
