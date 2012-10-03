function outputHistoryToFile(history, filename_prefix, objective)

% struct to vectors
iter = cat(1, history.iteration);
obj = cat(1, history.obj);
obj_qp = cat(1, history.obj_qp);
obj_lp = cat(1, history.obj_lp);
obj_decoded = cat(1, history.obj_decoded);
rho = cat(1, history.rho);

% output
header='iteration, obj, obj_qp, obj_lp, obj_decoded, rho';
filename = [filename_prefix '.csv'];
outid = fopen(filename, 'w');
fprintf(outid, '%s\n', header);
fclose(outid);
csv_data = [cumsum(iter(:)) obj(:) obj_qp(:) obj_lp(:) obj_decoded(:) rho(:)];
dlmwrite (filename, csv_data, '-append');

% output lines where rho changes
filename = [filename_prefix '_rho.tikz.tex'];
fid = fopen(filename, 'w');
idx = find(diff(rho)>eps);
c = cumsum(iter);
for i=1:numel(idx)
    x = c(idx(i));
    fprintf(fid, '\\draw[lightgray,dashed,-,thin] (axis cs:%d,\\ymin) -- (axis cs:%d,\\ymax);\n', x, x);
    %node [anchor=west,black] at (axis cs:1000,-1500) {\tiny$\rho=0.2$};');')
end
fclose(fid);

% additional horizontal lines (solutions by competitors)
if (nargin > 2)
    for idx=1:numel(objective)
        filename = [filename_prefix '_' objective(idx).name '.tikz.tex'];
        fid = fopen(filename, 'w');
        y = objective(idx).value;
        fprintf(fid, '\\draw[%s,dashed,-,thin] (axis cs:\\xmin,%d) -- (axis cs:\\xmax,%d);\n', objective(idx).color, y, y);
        fclose(fid);
    end
end
