function e = computeEntropy(mu)

e = 0;
if (~iscell(mu))
    for idx=1:size(mu,2)
        m = mu(:,idx);
        mt = m+eps;
        e = e - sum(m(:).*log(mt(:)));
    end
else
    for idx=1:size(mu,2)
        m = mu{idx};
        mt = m+eps;
        e = e - sum(m(:).*log(mt(:)));
    end
end
