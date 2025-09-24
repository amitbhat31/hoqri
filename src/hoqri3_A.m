function [U,orth_obj,recon_obj, tim] = hoqri3_A(X,K,U, Xn, iters, tol)

tic;

N = ndims(X);
normX = Xn;

I = X.size;
if isa(X,'sptensor')
    subs = X.subs;
    vals = X.vals;
elseif strcmp(getstructure(X),'sparse')
    subs = X.sub;
    vals = X.val;
end

reschangetol = tol;
recon_err_old = inf;

U1 = U{1};
U2 = U{2};
U3 = U{3};

orth_obj = [];
recon_obj = [];
tim = [];


fprintf('      ');
for itr = 1:iters

    fprintf('\b\b\b\b\b\b%6i',itr);
    
    Gnorm2 = 0;
    
    % U1
    M2 = U2(subs(:, 2), :);
    M3 = U3(subs(:, 3), :);

    A1 = zeros(I(1), K(1));
    for k2 = 1:K(2)
        v2 = vals .* M2(:, k2); %nnz x 1
        y_int = v2 .* M3;  %nnz x K_3
        Y = zeros(I(1), K(3));
        for k3 = 1:K(3)
            Y(:, k3) = accumarray(subs(:, 1), y_int(:, k3), [I(1),1]);
        end
        G = Y' * U1;
        A1 = A1 + Y*G;
        Gnorm2 = Gnorm2 + sum(G(:).^2);
    end
    [U1, ~] = qr(A1, 0);
    
    % U2
    M1 = U1(subs(:, 1), :);
    M3 = U3(subs(:, 3), :);

    A2 = zeros(I(2), K(2));
    for k1 = 1:K(1)
        v1 = vals .* M1(:, k1); %nnz x 1
        y_int = v1 .* M3;  %nnz x K_3
        Y = zeros(I(2), K(3));
        for k3 = 1:K(3)
            Y(:, k3) = accumarray(subs(:, 2), y_int(:, k3), [I(2),1]);
        end
        G = Y' * U2;
        A2 = A2 + Y*G;
    end
    [U2, ~] = qr(A2, 0);
    
    % U3
    M1 = U1(subs(:, 1), :);
    M2 = U2(subs(:, 2), :);

    A3 = zeros(I(3), K(3));
    for k1 = 1:K(1)
        v1 = vals .* M1(:, k1); %nnz x 1
        y_int = v1 .* M2;  %nnz x K_2
        Y = zeros(I(3), K(2));
        for k2 = 1:K(2)
            Y(:, k2) = accumarray(subs(:, 3), y_int(:, k2), [I(3),1]);
        end
        G = Y' * U3;
        A3 = A3 + Y*G;
    end
    [U3, ~] = qr(A3, 0);
    
    % core = ttm(X, {U1', U2', U3'});
    % X_pred = ttm(core, {U1, U2, U3});
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