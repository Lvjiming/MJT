function I_denoised = dictionary_learning(I)
I = double(I);
I = I / max(I(:));

X = I(:);

K = 256; 
D = randn(numel(I), K);
D = normc(D);
T = 10; 
for t = 1:T
    alpha = OMP(D, X, 10);
    for k = 1:K
        idx = find(alpha(k, :) ~= 0);
        if isempty(idx)
            continue;
        end
        D(:, k) = 0;
        R = X - D * alpha;
        E = R(:, idx) + D(:, k) * alpha(k, idx);
        [U, S, V] = svds(E, 1);
        D(:, k) = U;
        alpha(k, idx) = S * V';
    end
end

alpha = OMP(D, X, 11);
X_rec = D * alpha;
I_rec = reshape(X_rec, size(I));
I_denoised = I_rec;
end

function alpha = OMP(D, X, k)

N = size(D, 2);
r = X;
idx = [];
alpha = zeros(N, 1);
for i = 1:k
    proj = D' * r;
    [~, j] = max(abs(proj));
    idx = [idx, j];
    alpha(idx) = pinv(D(:, idx)) * X;
    r = X - D(:, idx) * alpha(idx);
end
end