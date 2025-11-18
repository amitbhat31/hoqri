clear 
close all
clc

core_list = [4:2:20];
len_core = length(core_list);

rank = 600;
nnz = 10;
algo = "lmlra_nls";

iter_list = [];
obj_list = [];
time_list = [];
try
    for rank_ind = 1:len_core
        core_rank = core_list(rank_ind);
    
        I = rank * [1, 1, 1];
        K = core_rank * [1, 1, 1];
    
        X = sptenrand(I, nnz*rank);
        Xn = norm(X)^2;
    
        init = {orth(randn(I(1),K(1))),orth(randn(I(2),K(2))),orth(randn(I(3),K(3)))};

        if algo == "hoqri"
            [U, obj, time] = hoqri_3ways(X, K, init);
        elseif algo == "lmlra_hooi"
            X = double(X);
            [~,S,output_hooi] = lmlra_hooi_time(X,init);
            time = output_hooi.itertime;
            obj = output_hooi.normS;
            disp(core_rank);
        elseif algo == "lmlra_nls"
            X = double(X);
            tic;
            [~,S,output_nls] = lmlra_nls(X,init, rand(K));
            iter_time = toc;
            time = [iter_time];
            obj = output_nls.fval;
            disp(core_rank);
        elseif algo == "lmlra_minf"
            X = double(X);
            tic;
            [~,S,output_minf] = lmlra_minf(X,init, rand(K));
            iter_time = toc;
            time = [iter_time];
            obj = output_minf.fval;
            disp(core_rank);
        else
            [T,Uinit,obj,time] = tucker_als_time(X,K,'tol', 1e-10, 'maxiters',100, 'printitn',0 );
            disp(core_rank)
        end
    
        iter_list(end + 1) = length(obj);
        obj_list(end+1) = obj(end);
        time_list(end+1) = time(end);
    end
    
    cores = core_list(:);
    iterations = iter_list(:);
    objective = obj_list(:);
    time = time_list(:);
    
    results_table = table(cores, iterations, objective, time);
    
    filename = sprintf('fig6_%s_dim%d_nnz%d_results.csv', algo, rank, nnz);
    
    writetable(results_table, filename);
    
    fprintf('Data successfully saved to %s\n', filename);
catch ME
    fprintf('\nAn error occurred: %s\n', ME.message);

    num_completed = length(iter_list);
    cores = core_list(1:num_completed)'; 
    
    iterations = iter_list(:);
    objective = obj_list(:);
    time = time_list(:);
   
    if ~isempty(dims)
        results_table = table(cores, iterations, objective, time);
        
        filename = sprintf('fig5_%s_dim%d_nnz%d_results.csv', algo, rank, nnz);
        
        writetable(results_table, filename);
        fprintf('Partial data successfully saved to %s\n', filename);
    else
        fprintf('No data was processed before the error occurred. No file saved.\n');
    end
    rethrow(ME);
end




