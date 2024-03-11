%Computing low-frequency a priori information
function  A=Lowfrequencyprior(c,m,n)
c=im2double(c);
[a,b,~]=size(c);
R1=c(:,:,1);
G1=c(:,:,2);
B1=c(:,:,3);
A=zeros(a,b);
d=A;

for i=1:a
    for j=1:b
        d(i,j)=min(R1(i,j),G1(i,j));
        A(i,j)=min(d(i,j),B1(i,j));
    end
end 
A=ordfilt2(A,1,ones(m,n));                     % Minimum value filtering

