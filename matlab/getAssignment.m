function assign = getAssignment(mu)

if (~iscell(mu))
    [dummy, x] = max(mu, [], 1);
    assign = x;
else
    for i=1:numel(mu)
        [dummy, x] = max(mu{i});
        assign(i) = x;
    end
end
