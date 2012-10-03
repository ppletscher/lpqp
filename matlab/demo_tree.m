% dimensionality of the problem
m = 2;
n = 2;
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

% run tree solver for each decomposition
edges = graph-1;
V = V(:,decomposition{1}+1);
edges = edges(:,decomposition{1}+1);
[logz, mu_unary, mu_pair] = mex_treeinference(-D, -V, edges, 1.0);
