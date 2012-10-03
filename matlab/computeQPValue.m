function energy = computeQPValue(mu_unary, theta_unary, theta_pair, edges)

if (~iscell(mu_unary))
    U = mu_unary.*theta_unary;
    energy = sum(U(:));
    
    num_edges = size(edges,2);
    for idx=1:num_edges
        mu_pair = mu_unary(:,edges(1,idx))*(mu_unary(:,edges(2,idx))');
        energy = energy + sum(theta_pair(:,idx).*mu_pair(:));
    end
else
    energy = 0;
    for idx=1:numel(mu_unary)
        energy = energy + mu_unary{idx}'*theta_unary{idx};
    end
    
    num_edges = size(edges,2);
    for idx=1:num_edges
        mu_pair = mu_unary{edges(1,idx)}*(mu_unary{edges(2,idx)}');
        energy = energy + theta_pair{idx}'*mu_pair(:);
    end
end

end % computeQPValue
