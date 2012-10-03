% dimensionality of the problem
m = 50;
n = 50;
nStates = 2;

rand('seed', 0);
randn('seed', 0);

% setup problem
[graph, decomposition] = generateGridGraph(m,n);
graph = graph';

r = randn(1, m*n);
D = repmat([-1 1]', [1 m*n]).*repmat(r, [2 1]);

%r = randn(1,size(graph,2)); % LP probably not tight
r = rand(1,size(graph,2)); % LP should be tight
V = repmat([1 -1 -1 1]', [1 size(graph,2)]).*repmat(r, [4 1]);

% run lpqp solver
options = [];
%options.solver = 'fistadescent';
%options.solver = 'lbfgs';
options.rho = 1e-5;
options.num_max_iter = 2000;
options.eps_gnorm = 1e-10;
mu_unary_sdd = mex_smoothdualdecomposition(D, V, graph-1, decomposition, options);
val_qp_sdd = computeQPValue(mu_unary_sdd, D, V, graph);

%pause

options = [];
options.num_max_iter = 50;
mu_unary_lpqp = mex_trws(D, V, graph-1, options);
val_qp_lpqp = computeQPValue(mu_unary_lpqp, D, V, graph);

T = mu_unary_lpqp-mu_unary_sdd;
imagesc(reshape(T(1,:), [m n]))

%val_qp_sdd
%val_qp_lpqp
