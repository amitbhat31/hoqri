clear 
close all
clc

I_list = [100:100:1000, 1200:200:2000, 2500:500:3000];
len_I = length(I_list);

core_rank = 10;
nnz = 100;
algo = "lmlra_minf";

iter_list = [];
obj_list = [];
time_list = [];

try
    for rank_ind = 1:len_I
        rank = I_list(rank_ind);
    
        I = rank * [1, 1, 1];
        K = core_rank * [1, 1, 1];
    
        X = sptenrand(I, nnz*rank);
        Xn = norm(X)^2;
    
        init = {orth(randn(I(1),K(1))),orth(randn(I(2),K(2))),orth(randn(I(3),K(3)))};

        if algo == "hoqri"
            [U, obj, time] = hoqri3_A(X, K, init);
        elseif algo == "lmlra_hooi"
            % T.val = X.vals;
            % T.sub = X.subs;
            % T.size = I;
            % T.sparse = true;
            % T = fmt(T);
            % [U, obj, time] = hooi_tensorlab(T, K, init);

            X = double(X);
            [~,S,output_hooi] = lmlra_hooi_time(X,init);
            time = output_hooi.itertime;
            obj = output_hooi.normS;
            disp(rank);
        elseif algo == "lmlra_nls"
            X = double(X);
            tic;
            [~,S,output_nls] = lmlra_nls(X,init, rand(K));
            iter_time = toc;
            time = [iter_time];
            obj = output_nls.fval;
            disp(rank);
        elseif algo == "lmlra_minf"
            X = double(X);
            tic;
            [~,S,output_minf] = lmlra_minf(X,init, rand(K));
            iter_time = toc;
            time = [iter_time];
            obj = output_minf.fval;
            disp(rank);
        else
            [T,Uinit,obj,time] = tucker_als_time(X,K,'tol', 1e-10, 'maxiters',100, 'printitn',0 );
            disp(rank)
        end
    
        iter_list(end + 1) = length(obj);
        obj_list(end+1) = obj(end);
        time_list(end+1) = time(end);
    end
    
    dims = I_list(:);
    iterations = iter_list(:);
    objective = obj_list(:);
    time = time_list(:);
    
    results_table = table(dims, iterations, objective, time);
    
    % 3. Define the filename for your CSV.
    filename = sprintf('fig5_%s_rank%d_nnz%d_results.csv', algo, core_rank, nnz);
    
    % 4. Write the table to the CSV file.
    writetable(results_table, filename);
    
    % Display a confirmation message in the command window.
    fprintf('Data successfully saved to %s\n', filename);
catch ME
    fprintf('\nAn error occurred: %s\n', ME.message);

    num_completed = length(iter_list);
    dims = I_list(1:num_completed)'; % Ensure it's a column vector
    
    iterations = iter_list(:);
    objective = obj_list(:);
    time = time_list(:);
   
    if ~isempty(dims)
        results_table = table(dims, iterations, objective, time);
        
        filename = sprintf('fig5_%s_rank%d_nnz%d_results.csv', algo, core_rank, nnz);
        
        writetable(results_table, filename);
        fprintf('Partial data successfully saved to %s\n', filename);
    else
        fprintf('No data was processed before the error occurred. No file saved.\n');
    end
    rethrow(ME);
end





