% dimensionality of the problem
m = 30;
n = 30;
nStates = 2;

rand('seed', 0);
randn('seed', 0);

% setup problem
[graph, decomposition] = generateGridGraph(m,n);
graph = graph';

r = randn(1, m*n);
D = repmat([-1 1]', [1 m*n]).*repmat(r, [2 1]);

r = 2*randn(1,size(graph,2)); % LP probably not tight
%r = rand(1,size(graph,2)); % LP should be tight
V = repmat([1 -1 -1 1]', [1 size(graph,2)]).*repmat(r, [4 1]);


% TRWS
options = [];
options.num_max_iter = 50;
mu_unary_trws = mex_trws(D, V, graph-1, options);

mu_unary_trws = roundSolution(mu_unary_trws);
val_trws = computeQPValue(mu_unary_trws, D, V, graph);


% run lpqp solver (smooth dual decomposition version)
options = [];
options.rho_start = 5e0;
%options.rho_end = 1e2;
%options.num_max_iter_dc = 20;
%options.num_max_iter_mp = 300;

options.solver_sdd = 'lbfgs';
[mu_unary_sdd_lbfgs, history_sdd_lbfgs] = mex_lpqp(D, V, graph-1, options, decomposition);

options.solver_sdd = 'fistadescent';
[mu_unary_sdd, history_sdd] = mex_lpqp(D, V, graph-1, options, decomposition);

mu_unary_sdd = roundSolution(mu_unary_sdd);
val_sdd = computeQPValue(mu_unary_sdd, D, V, graph);


% run lpqp solver (norm-product BP version)
[mu_unary_npbp, history_npbp] = mex_lpqp(D, V, graph-1);

mu_unary_npbp = roundSolution(mu_unary_npbp);
val_npbp = computeQPValue(mu_unary_npbp, D, V, graph);
