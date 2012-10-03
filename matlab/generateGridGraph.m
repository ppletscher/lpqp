function [graph, decomposition] = generateGridGraph(m,n)

num_edges = m*(n-1) + (m-1)*n;

graph = zeros(num_edges,2);
decomposition = cell(2,1);
decomposition{1} = zeros((m-1)*n,1);
decomposition{2} = zeros(m*(n-1),1);

vert_idx = 0;
hor_idx = 0;
e_idx = 0;
for j=1:n
    for i=1:m
        if (i<m)
            graph(e_idx+1,:) = [i+(j-1)*m i+1+(j-1)*m];
            decomposition{1}(vert_idx+1) = e_idx;
            vert_idx = vert_idx+1;
            e_idx = e_idx+1;
        end
        if (j<n)
            graph(e_idx+1,:) = [i+(j-1)*m i+(j)*m];
            decomposition{2}(hor_idx+1) = e_idx;
            hor_idx = hor_idx+1;
            e_idx = e_idx+1;
        end
    end
end
