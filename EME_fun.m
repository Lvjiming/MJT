function EME=EME_fun(I,L)
[height,width]=size(I);

%使用L*L大小的块对M*M大小的图像进行运算
how_many=floor(height/L);
E=0;
B1=zeros(L);
m1=1;
for m=1:how_many
    n1=1;
    for n=1:how_many
        B1=I(m1:m1+L-1,n1:n1+L-1);
        b_min=min(min(B1));
        b_max=max(max(B1));
        
        if b_min > 0
            % b_ratio = b_max / b_min;
            b_ratio = double(b_max) / double(b_min); % The b_ratio must be a floating point number.
            % disp(['b_max = ', num2str(b_max), ', b_min = ', num2str(b_min), ', b_ratio = ', num2str(b_ratio)]);
            E = E + 20 * log10(b_ratio);
        end       
        n1=n1+L;
    end
    m1=m1+L;
end
E=(E/how_many)/how_many;
EME=E;

end