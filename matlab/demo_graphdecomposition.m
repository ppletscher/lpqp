% dimensionality of the problem
m = 10;
n = 10;

% setup problem
[graph, decomposition_manual] = generateGridGraph(m,n);
graph = graph';

decomposition = mex_graphdecomposition(graph-1);

r = randn(1, m*n);
D = repmat([-1 1]', [1 m*n]).*repmat(r, [2 1]);

r = 2*randn(1,size(graph,2)); % LP probably not tight
%r = rand(1,size(graph,2)); % LP should be tight
V = repmat([1 -1 -1 1]', [1 size(graph,2)]).*repmat(r, [4 1]);

% run LPQPSDD solver with automatic decomposition
options = struct([]);
mu_unary = mex_lpqp(D, V, graph-1, options, decomposition);
val_automatic = computeQPValue(mu_unary, D, V, graph)

% run LPQPSDD solver with manual decomposition
mu_unary = mex_lpqp(D, V, graph-1, options, decomposition_manual);
val_manual = computeQPValue(mu_unary, D, V, graph)
