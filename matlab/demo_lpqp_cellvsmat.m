% dimensionality of the problem
m = 20;
n = 20;
nStates = 2;

% setup problem
graph = generateGridGraph(m,n)';

r = randn(1, m*n);
D = repmat([-1 1]', [1 m*n]).*repmat(r, [2 1]);

r = randn(1,size(graph,2));
V = repmat([1 -1 -1 1]', [1 size(graph,2)]).*repmat(r, [4 1]);

% run lpqp solver
options = [];
options.rho_start = 1e-1;
mu_unary = mex_lpqp(D, V, graph-1, options);
val_qp = computeQPValue(mu_unary, D, V, graph)

% round solution
mu_unary = roundSolution(mu_unary);
val_qp = computeQPValue(mu_unary, D, V, graph)

% run lpqp solver for cell arrays
Dtemp = mat2cell(D, size(D,1), repmat(1,[1 size(D,2)]));
Vtemp = mat2cell(V, size(V,1), repmat(1,[1 size(V,2)]));
mu_unary = mex_lpqp(Dtemp, Vtemp, graph-1, options);
