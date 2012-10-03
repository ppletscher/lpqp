#include <GraphDecomposition.h>
#include <stdio.h>
#include <vector>
#include <set>
#include <map>
#include <stack>
#include <queue>
#include <cmath>
#include <Eigen/Core>

GraphDecomposition::GraphDecomposition(Eigen::MatrixXi& edges):
    _edges(edges)
{
    _num_edges = edges.cols();

    // number of vertices
    _num_vertices = 0;
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = edges(0, e_idx);
        size_t j = edges(1, e_idx);
        _num_vertices = std::max(_num_vertices, i);
        _num_vertices = std::max(_num_vertices, j);
    }
    _num_vertices++;

    // initialize the graph data structure from the edges
    _neighbors.resize(_num_vertices);
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = edges(0, e_idx);
        size_t j = edges(1, e_idx);

        _neighbors[i].push_back(std::make_pair(j, e_idx));
        _neighbors[j].push_back(std::make_pair(i, e_idx));
    }
}

void GraphDecomposition::pprint(std::vector< std::vector<size_t> > &decomposition)
{
	printf("%d trees in the decomposition\n",static_cast<int>(decomposition.size()));
	std::map<int, int> num_edges_in_trees;
	for (size_t d_idx=0; d_idx<decomposition.size(); d_idx++) {

	    printf("Tree %d has %d edges:\n",static_cast<int>(d_idx),static_cast<int>(decomposition[d_idx].size()));

		if (num_edges_in_trees.find(decomposition[d_idx].size()) == num_edges_in_trees.end()) {
		    num_edges_in_trees[decomposition[d_idx].size()] = 1;
		}
		else {
		    num_edges_in_trees[decomposition[d_idx].size()] += 1;
		}


		//for (size_t e_idx=0; e_idx<decomposition[d_idx].size(); e_idx++) {
		//    size_t edge_id = decomposition[d_idx][e_idx];
		//    printf("edge %u from %u to %u\n",edge_id, _edges(0,edge_id),_edges(1,edge_id));
		//}
	}

	std::map<int,int>::iterator it = num_edges_in_trees.begin();
	for (; it != num_edges_in_trees.end(); it++ ) {
	    //std::cout << (*it).second << " trees with " << (*it).first << " edges " << std::endl;
	    printf("%d trees with %d edges.\n", (*it).second, (*it).first);
    }
}

std::vector< std::vector<size_t> > GraphDecomposition::decompose(DecompositionType type)
{
    switch (type){
        case DECOMPOSITION_STACK:
            return decompose_with_stack();
        case DECOMPOSITION_QUEUE:
            return decompose_with_queue();
        default:
            return decompose_with_stack();
    }
}

std::vector< std::vector<size_t> > GraphDecomposition::decompose_with_stack()
{
    std::vector< std::vector<size_t> > decomposition;
    std::vector<bool> edge_covered;
    edge_covered.resize(_num_edges, false);
    size_t num_edge_covered = 0;
    std::set<size_t> vertex_uncovered;
    std::vector<size_t> vertex_coverage_counter;
    vertex_coverage_counter.resize(_num_vertices);
    for (size_t v_idx=0; v_idx<_num_vertices; v_idx++) {
        if (_neighbors[v_idx].size() > 0) {
            vertex_uncovered.insert(v_idx);
        }
        vertex_coverage_counter[v_idx] = _neighbors[v_idx].size();
    }

    while (num_edge_covered < _num_edges) {
        std::vector<size_t> decomposition_current;
        std::vector<bool> node_in_current_decomposition;
        node_in_current_decomposition.resize(_num_vertices, false);

        //std::cout << vertex_uncovered.size() << std::endl;
        assert(vertex_uncovered.size() > 0);

        size_t head = *(vertex_uncovered.begin());
        std::stack< std::pair<size_t, size_t> > candidates; // edge index and new node, thats 1 of the differences between a stack and a queue

        // schedule first one
        for (size_t n_idx=0; n_idx<_neighbors[head].size(); n_idx++) {
            size_t e_idx = _neighbors[head][n_idx].second;
            size_t j = _neighbors[head][n_idx].first;
            if (!edge_covered[e_idx]) {
                candidates.push( std::make_pair(e_idx, j) );
            }
        }

        // go through the stack
        while (!candidates.empty()) {
            std::pair<size_t, size_t> p = candidates.top();

            candidates.pop();
            size_t v_to = p.second;
            size_t e_idx = p.first;
            size_t v_from = _edges(0,e_idx);
            if (static_cast<size_t>(_edges(0,e_idx)) == v_to) {
                v_from = _edges(1,e_idx);
            }

            if (!edge_covered[e_idx] && !node_in_current_decomposition[v_to]) {
                // edge is safe to add, update datastructures
                decomposition_current.push_back(e_idx);
                edge_covered[e_idx] = true;
                node_in_current_decomposition[v_from] = true;
                node_in_current_decomposition[v_to] = true;
                assert(vertex_coverage_counter[v_to] > 0);
                vertex_coverage_counter[v_to]--;
                assert(vertex_coverage_counter[v_from] > 0);
                vertex_coverage_counter[v_from]--;
                num_edge_covered++;

                //std::cout << "covered edge "  << e_idx << " i: " << v_from <<
                //" j: " << v_to << std::endl;

                // add neighbors of v_to to stack
                for (size_t n_idx=0; n_idx<_neighbors[v_to].size(); n_idx++) {
                    size_t e_idx_n = _neighbors[v_to][n_idx].second;
                    size_t j = _neighbors[v_to][n_idx].first;
                    if (!edge_covered[e_idx_n]) {
                        candidates.push( std::make_pair(e_idx_n, j) );
                    }
                }

                // update vertex_uncovered
                if (vertex_coverage_counter[v_from] == 0) {
                    vertex_uncovered.erase(v_from);
                }
                if (vertex_coverage_counter[v_to] == 0) {
                    vertex_uncovered.erase(v_to);
                }
            }
        }

        decomposition.push_back(decomposition_current);
    }

    return decomposition;
}

std::vector< std::vector<size_t> > GraphDecomposition::decompose_with_queue()
{
    std::vector< std::vector<size_t> > decomposition;
    std::vector<bool> edge_covered;
    edge_covered.resize(_num_edges, false);
    size_t num_edge_covered = 0;
    std::set<size_t> vertex_uncovered;
    std::vector<size_t> vertex_coverage_counter;
    vertex_coverage_counter.resize(_num_vertices);
    for (size_t v_idx=0; v_idx<_num_vertices; v_idx++) {
        if (_neighbors[v_idx].size() > 0) {
            vertex_uncovered.insert(v_idx);
        }
        vertex_coverage_counter[v_idx] = _neighbors[v_idx].size();
    }

    while (num_edge_covered < _num_edges) {
        std::vector<size_t> decomposition_current;
        std::vector<bool> node_in_current_decomposition;
        node_in_current_decomposition.resize(_num_vertices, false);

        //std::cout << vertex_uncovered.size() << std::endl;
        assert(vertex_uncovered.size() > 0);

        size_t head = *(vertex_uncovered.begin());
        std::queue< std::pair<size_t, size_t> > candidates; // 1 of the differences between a stack and a queue

        // schedule first one
        for (size_t n_idx=0; n_idx<_neighbors[head].size(); n_idx++) {
            size_t e_idx = _neighbors[head][n_idx].second;
            size_t j = _neighbors[head][n_idx].first;
            if (!edge_covered[e_idx]) {
                candidates.push( std::make_pair(e_idx, j) );
            }
        }

        // go through the stack
        while (!candidates.empty()) {

            std::pair<size_t, size_t> p = candidates.front(); // 1 of the differences between a stack and a queue

            candidates.pop();
            size_t v_to = p.second;
            size_t e_idx = p.first;
            size_t v_from = _edges(0,e_idx);
            if (static_cast<size_t>(_edges(0,e_idx)) == v_to) {
                v_from = _edges(1,e_idx);
            }

            if (!edge_covered[e_idx] && !node_in_current_decomposition[v_to]) {
                // edge is safe to add, update datastructures
                decomposition_current.push_back(e_idx);
                edge_covered[e_idx] = true;
                node_in_current_decomposition[v_from] = true;
                node_in_current_decomposition[v_to] = true;
                assert(vertex_coverage_counter[v_to] > 0);
                vertex_coverage_counter[v_to]--;
                assert(vertex_coverage_counter[v_from] > 0);
                vertex_coverage_counter[v_from]--;
                num_edge_covered++;

                //std::cout << "covered edge "  << e_idx << " i: " << v_from <<
                //" j: " << v_to << std::endl;

                // add neighbors of v_to to stack
                for (size_t n_idx=0; n_idx<_neighbors[v_to].size(); n_idx++) {
                    size_t e_idx_n = _neighbors[v_to][n_idx].second;
                    size_t j = _neighbors[v_to][n_idx].first;
                    if (!edge_covered[e_idx_n]) {
                        candidates.push( std::make_pair(e_idx_n, j) );
                    }
                }

                // update vertex_uncovered
                if (vertex_coverage_counter[v_from] == 0) {
                    vertex_uncovered.erase(v_from);
                }
                if (vertex_coverage_counter[v_to] == 0) {
                    vertex_uncovered.erase(v_to);
                }
            }
        }

        decomposition.push_back(decomposition_current);
    }

    return decomposition;
}
