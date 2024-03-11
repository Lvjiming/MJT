function guass=Guass_despeckle(I)

[M,N]=size(I(:,:,1));
J = fft2(I);
I_1channel=I(:,:,1);
I_1D=I_1channel(:); 
W = abs(J/255);
W = fftshift(W);
K = fftshift(J);
d0=30;  
hGus_l = zeros(M,N);
m_mid=floor(M/2);
n_mid=floor(N/2);
for i = 1:M
    for j = 1:N
        d = ((i-m_mid)^2+(j-n_mid)^2); 
        hGus_l(i,j) =exp(-(d)/(2*(d0^2)));  
    end
end
img_lpf = hGus_l.*K; 
img_lpf = ifftshift(img_lpf);
Q = ifft2(img_lpf);
guass=abs(Q);

end