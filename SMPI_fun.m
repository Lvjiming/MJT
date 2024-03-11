function smpi=SMPI_fun(imnoise,imde)
Q=1+abs(mean2(imnoise)-mean2(imde));
hou=std2(imde)/(std2(imnoise));
smpi=Q.*hou;
if isnan(smpi)
    smpi=0.1;
end
