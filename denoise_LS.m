function output = denoise_LS(input, window_size)
[m, n] = size(input);
output = input;

for i = 1:m
    for j = 1:n
        window = input(max(i-floor(window_size/2),1):min(i+floor(window_size/2),m), ...
            max(j-floor(window_size/2),1):min(j+floor(window_size/2),n));
        
        mean_val = mean(window(:));
        std_val = std(single(window(:))); % 将 window 转换为单精度类型再计算标准差
        
        if abs(input(i,j) - mean_val) > 0.05*std_val
            output(i,j) = mean(input(max(i-1,1):min(i+1,m), max(j-1,1):min(j+1,n)),'all');
        end
    end
end

end