close all;
clc

Path = image-path; 
paths={[Path,'1ruilifenbu.png'],[Path,'2farmnakagami.png'],[Path,'3Kforest.png'],[Path,'4Lognormalcity.png'],[Path,'5mountainareaFisher.png'],[Path,'6seafacegam.png']};
% Loop through each path
for i = 1:length(paths)
    path = paths{i};
    [~, filename, ext] = fileparts(path); % Takes the last filename of the path
    I1=imread(path);
    if numel(size(I1)) == 2 %Determine whether it is single channel, single channel will copy the image into three channels
        I1 = repmat(I1, [1 1 3]);
    end

    A1=Lowfrequencyprior(I1,5,5);
    A1_guide=imguidedfilter(A1);   

    add_noise=max(max(A1)); 
    Ac1=[add_noise,add_noise,add_noise];

    img = double(I1) / 255;
    grayImg = rgb2gray(img);
    grayImg = im2double(grayImg);
    % Expand the grey scale image into a one dimensional array
    grayImg = grayImg(:);

    x = linspace(0, 1, length(grayImg))';
    p = polyfit(x, grayImg, 1);   % Calculate regression values
    yfit = polyval(p, x); % y-axis represents fitted regression values, x-axis represents image values

     % Calculate the value of y when x = 0
    y0 = polyval(p, 0);
    y1=polyval(p, 1);
    scatter(x, grayImg, 5, 'filled');
    hold on;
    plot(x, yfit, 'r-', 'LineWidth', 1.5);

     % shows the regression value and the y-value when x = 0
    text(0.7, 0.9, ['slope = ', num2str(p(1))], 'Units', 'normalized');
    text(0.7, 0.8, ['y(x=0) = ', num2str(y0)], 'Units', 'normalized');

    title('Scatter Plot with Regression Line');
    xlabel('X');
    ylabel('Y');

    w=1-(y0+y1)/2;

    [t1,R1,G1,B1]=grayfactor(I1,Ac1,w);
    I2=cat(3,R1,G1,B1);
    guass_im=Guass_despeckle(I1);
    wave_im=Wave_despeckle(I1);
    dictionary_im = dictionary_learning(I1);
    LS_im = denoise_LS(I1, 5); 
    
    save_path1 = sprintf(save_path, filename);
    save_path2 = sprintf(save_path, filename);
    save_path3= sprintf(save_path, filename);
    imwrite(I1, save_path1);
    imwrite(A1, save_path2);
    imwrite(abs(I2), save_path3);
    
    save_path4= sprintf(save_path, filename);
    save_path5= sprintf(save_path, filename);
    save_path6= sprintf(save_path, filename);
    save_path7= sprintf(save_path, filename);
    imwrite(abs(guass_im/255), save_path4);
    imwrite(abs(wave_im/255), save_path5);
    imwrite(abs(dictionary_im), save_path6);
    imwrite(abs(LS_im), save_path7);
    
    figure(1);
    subplot(241); imshow(I1); title('original image');
    subplot(242); imshow(A1); title('Lowf requency priorimage');
    subplot(243); imshow(abs(I2)); title('Denoising Image Enhancement');
    subplot(244); imshow(guass_im/255); title('Denoising Image Gaussian');
    subplot(245); imshow(wave_im/255); title('Denoising Image wave');
    subplot(246); imshow(dictionary_im); title('Denoising Image dl');
    subplot(247); imshow(LS_im); title('Denoising Image ls'); 

    qu_smpi=SMPI_fun(I1,I2);
    qu_eme=EME_fun(abs(I2(:,:,1)).*255,8);

    gus_smpi=SMPI_fun(I1,guass_im);
    gus_eme=EME_fun(guass_im(:,:,1),8);

    smpi_value_wave=SMPI_fun(I1,wave_im);
    eme_value_wave=EME_fun(wave_im(:,:,1),8);

    smpi_value_dl=SMPI_fun(I1,dictionary_im.*255);
    eme_value_dl=EME_fun(dictionary_im(:,:,1),8);

    smpi_value_LS=SMPI_fun(I1,LS_im);
    eme_value_LS=EME_fun(LS_im(:,:,1),8);

    disp([filename,sprintf('\t'),'SMPI_Our: ', num2str(qu_smpi),sprintf('\t'), 'EME_Our: ', num2str(qu_eme)])
    disp([filename,sprintf('\t'),'SMPI_gaussian: ', num2str(gus_smpi),sprintf('\t'), 'EME_gaussian: ', num2str(gus_eme)]) 
    disp([filename,sprintf('\t'),'SMPI_Wavelet: ', num2str(smpi_value_wave),sprintf('\t'), 'EME_Wavelet: ', num2str(eme_value_wave)])
    disp([filename,sprintf('\t'),'SMPI_DL: ', num2str(smpi_value_dl), sprintf('\t'),'EME_DL: ', num2str(eme_value_dl)]) 
    disp([filename,sprintf('\t'),'SMPI_LS: ', num2str(smpi_value_LS), sprintf('\t'),'EME_LS: ', num2str(eme_value_LS)])

    disp('finish')

end
