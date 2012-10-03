function dist = computeKLDivergence(graph, mu_unary, mu_pair)

dist = 0;
if (~iscell(mu_unary))
    for e_idx=1:size(graph,2)
        m1 = mu_unary(:,graph(1,e_idx(1)));
        m2 = mu_unary(:,graph(2,e_idx(1)));
        M = m1*m2';
        kl = kldiv(mu_pair(:,e_idx), M(:));
        dist = dist + kl;
    end
else
    for e_idx=1:size(graph,2)
        m1 = mu_unary{graph(1,e_idx(1))};
        m2 = mu_unary{graph(2,e_idx(1))};
        M = m1*m2';
        kl = kldiv(mu_pair{e_idx}, M(:));
        dist = dist + kl;
    end
end


function d = kldiv(p, q)

% TODO: improve this!

pt = p+eps;
qt = q+eps;
d = sum(p.*log(pt./qt));
if (d < 0)
    d = 0;
end
