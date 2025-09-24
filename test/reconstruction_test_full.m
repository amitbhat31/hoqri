clear;
close all
clc

seed = 15;

rank = 600;
core_rank = 10;
nnz = 100;
modes = 3;

iters = 100;
res_tolerance = 1e-10;

output_folder = 'recons_test';
filename = sprintf('fig%d_recons_all_rank%d_core%d_nnz%d_seed%d_results.csv', modes, rank, core_rank, nnz, seed);
full_path = fullfile(output_folder, filename);

algo_list = {'hoqri', 'lmlra_hooi', 'tucker_als', 'lmlra_minf', 'lmlra_nls'};

orth_obj = [];
recon_obj = [];
time = [];

rng(seed);

I = rank * ones(1, modes);
K = core_rank * ones(1, modes);

X = sptenrand(I, nnz*rank);
Xn = norm(X)^2;

init = cell(1, modes);
for i = 1:modes
    init{i} = orth(randn(I(i), K(i)));
end

for i = 1:5
    curr_algo = algo_list{i};
    fprintf('Running %s...\n', curr_algo);


    if curr_algo == "hoqri"

        if modes==3
            [U, orth_obj, recon_obj, time] = hoqri3_A(X, K, init, Xn, iters, res_tolerance);
        else
            [U, orth_obj, recon_obj, time] = hoqri3_N_A_opt(X, K, init, iters, res_tolerance);
        end
        disp(time(end));
    elseif curr_algo == "lmlra_hooi"
        X_dense = double(X);
        
        [~,S,output_hooi] = lmlra_hooi_time(X_dense,init, Xn);
        time = output_hooi.time;
        orth_obj = output_hooi.normS;
        recon_obj = output_hooi.normR;
        disp(time(end));
    elseif curr_algo == "lmlra_nls"
        X_dense = double(X);

        tic;
        [~,S,output_nls] = lmlra_nls(X_dense,init, rand(K));
        iter_time = toc;
        orth_obj = output_nls.fval;
        recon_obj = output_nls.fval;
        time = linspace(0, iter_time, length(recon_obj));
        disp(time(end));
    elseif curr_algo == "lmlra_minf"
        X_dense = double(X);

        tic;
        [~,S,output_minf] = lmlra_minf(X_dense,init, rand(K));
        iter_time = toc;
        orth_obj = output_minf.fval;
        recon_obj = output_minf.fval;
        time = linspace(0, iter_time, length(recon_obj));
        disp(time(end));
    else
        [T,Uinit,orth_obj, recon_obj, time] = tucker_als_time(X,K,'tol', res_tolerance, 'maxiters',iters, 'printitn',0 );
        disp(time(end));
    end

    orth_objective = orth_obj(:);
    recon_objective = recon_obj(:);
    time = time(:);

    orth_objective(end+1:iters+1) = NaN;
    recon_objective(end+1:iters+1) = NaN;
    time(end+1:iters+1) = NaN;

    curr_results_table = table(time, orth_objective, recon_objective);
    curr_results_table.Properties.VariableNames = {
        sprintf('%s_time', curr_algo), ...
        sprintf('%s_orth_obj', curr_algo), ...
        sprintf('%s_recon_obj', curr_algo)
    };
    
    if i == 1
        writetable(curr_results_table, full_path);
    else
        existing_table = readtable(full_path);
        combined_table = [existing_table, curr_results_table];
        writetable(combined_table, full_path);
    end
    fprintf('%s complete, data successfully saved to %s\n', curr_algo, filename);
end
