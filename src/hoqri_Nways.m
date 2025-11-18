function [U, orth_obj, recon_obj, tim] = hoqri_Nways(X, K, U, maxiter, tol)
% X: input tensor
% K: core tensor
% U: initial guesses for matrices
tic;

I = size(X);
N = length(I);
normX = norm(X)^2;

if isa(X, 'sptensor')
    subs = X.subs;
    vals = X.vals;
elseif strcmp(getstructure(X), 'sparse')
    subs = X.sub;
    vals = X.val;
end

reschangetol = tol;

U_subs = cell(1, N);
for m = 1:N
    U_subs{m} = U{m}(subs(:, m), :);
end

orth_obj = [];
recon_obj = [];
tim = [];

recon_err_old = inf;

fprintf('      ');
for itr = 1:maxiter
    fprintf('\b\b\b\b\b\b%6i',itr);
    
    Gnorm2 = 0;

    for n = 1:N 
        A_n = zeros(I(n), K(n));
        other_modes = [1:n-1, n+1:N];
        num_other_modes = length(other_modes);

        if num_other_modes == 0
            y = accumarray(subs(:, n), vals, [I(n), 1]);
            g = y' * U{n};
            A_n = y * g;

            if n == 1
                Gnorm2 = sum(g.^2);
            end
            [U{n}, ~] = qr(A_n, 0);
            continue;
        end

        K_other = K(other_modes);
        rank_indices = ones(1, num_other_modes);
        loop = true;

        while loop

            y_temp = vals;
            for i = 1:num_other_modes
                curr_mode = other_modes(i);
                curr_rank_idx = rank_indices(i);
                y_temp = y_temp .* U_subs{curr_mode}(:, curr_rank_idx);
            end

            y = accumarray(subs(:, n), y_temp, [I(n), 1]);
            g = y' * U{n};
            A_n = A_n + y * g;

            if n == 1
                Gnorm2 = Gnorm2 + sum(g.^2);
            end

            % while loop to mimic nested for loop structure
            d = num_other_modes;
            while d >= 1
                rank_indices(d) = rank_indices(d) + 1;
                if rank_indices(d) <= K_other(d)
                    break;
                else
                    rank_indices(d) = 1;
                    d = d - 1;
                end
            end

            if d < 1
                loop = false;
            end
        end

        [U{n}, ~] = qr(A_n, 0);
        U_subs{n} = U{n}(subs(:, n), :);

    end
    
    recon_err = abs(normX - Gnorm2);

    orth_obj = [ orth_obj, Gnorm2 ];
    recon_obj = [recon_obj, recon_err];
    tim = [tim, toc];

    reschange = abs(recon_err_old - recon_err);

    if (itr > 1) && (reschange < reschangetol)
        break;
    end

    recon_err_old = recon_err;
end
fprintf("\n")


