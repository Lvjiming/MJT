clc
clear
close all;

img_path='your path';
path=(img_path);
path_new=(img_path);

fileFolder1 = fullfile(path);
classname1 = dir(fileFolder1); % Note that it is the structure indexed with .XXX
num_img1=size(classname1);
num_img1=num_img1(1,1);% Takes the first value to be the total number of folders in a subclass

arr2_SMPI=[];
arr2_EME=[];

arr2_SMPI_gus=[];
arr2_EME_gus=[];

arr2_SMPI_wave=[];
arr2_EME_wave=[];

arr2_SMPI_dl=[];
arr2_EME_dl=[];

arr2_SMPI_LS=[];
arr2_EME_LS=[];

angle_arr=[]; %% Pitch angle name of the final output table
sub_class_arr=[]; %Final output of the table's subclass information

for ii=3:num_img1
    name_class=string(classname1(ii).name);% Get the name of the folder, but in string format.
    fileFolder = path+name_class; % "C:\Users\ljm\Desktop\march\class15"
    fileFolder2 = fullfile(fileFolder);
    classname2 = dir(fileFolder2); % This catalogue is now at the class level
    num_img2=size(classname2);
    num_img2=num_img2(1,1);%Take the first value as the total number of folders in a subcategory   
    
    for jj=3:num_img2
        name_class2=string(classname2(jj).name);% Get the name of a single image, but convert it to string format
        sub_class_arr=[sub_class_arr;name_class2];
        angle_arr=[angle_arr;name_class];
        path_img1=path+name_class+'\'+name_class2;
        fileFolder3 = fullfile(path_img1, '*.jpg');
        classname3 = dir(fileFolder3); % Note that it's the structure indexed with .XXX
        num_img=size(classname3);
        num_img=num_img(1,1);%Takes the first value to be the total number of folders in a subclass
        

        arr_SMPI=[];%arr, this is a tool to store the data values of all the images in each category one at a time.
        arr_EME=[];
        
        arr_SMPI_gus=[];
        arr_EME_gus=[];
        
        arr_SMPI_wave=[];
        arr_EME_wave=[];
        
        arr_SMPI_dl=[];
        arr_EME_dl=[];
        
        arr_SMPI_LS=[];
        arr_EME_LS=[];
            
        for mm=1:num_img
            name_img=string(classname3(mm).name);
            path_img2=path+name_class+'\'+name_class2+'\'+name_img;
            path_save_img=path_new+name_class+'\'+name_class2+'\'+name_img; 
            I1=imread(path_img2);
            
            if numel(size(I1)) == 2 %Determine whether it is single channel, single channel will copy the image into three channels
                I1 = repmat(I1, [1 1 3]);
            end
            
            A1=Lowfrequencyprior(I1,5,5);
            A1_guide=imguidedfilter(A1);      % guided filtering                
            add_noise=max(max(A1));
            Ac1=[add_noise,add_noise,add_noise];
            
            img = double(I1) / 255;
            grayImg = rgb2gray(img);
            grayImg = im2double(grayImg);
            grayImg = grayImg(:);
            x = linspace(0, 1, length(grayImg))';
            p = polyfit(x, grayImg, 1);
            yfit = polyval(p, x); % y-axis represents fitted regression values, x-axis represents image values

            y0 = polyval(p, 0);  % Calculate the value of y when x = 0
            y1=polyval(p, 1);
            w=1-(y0+y1)/2;
            
            [t1,R1,G1,B1]=grayfactor(I1,Ac1,w);
            I2=cat(3,R1,G1,B1);
            guass_im=Guass_despeckle(I1);
            wave_im=Wave_despeckle(I1);
            dictionary_im = dictionary_learning(I1);
            LS_im = denoise_LS(I1, 5);
            
            smpi_value=SMPI_fun(I1,I2);
            eme_value=EME_fun(abs(I2(:,:,1)).*255,8);
            
            smpi_value_gus=SMPI_fun(I1,guass_im);
            eme_value_gus=EME_fun(guass_im(:,:,1),8);
            
            smpi_value_wave=SMPI_fun(I1,wave_im);
            eme_value_wave=EME_fun(wave_im(:,:,1),8);
            
            smpi_value_dl=SMPI_fun(I1,dictionary_im.*255);
            eme_value_dl=EME_fun(dictionary_im(:,:,1),8);
            
            smpi_value_LS=SMPI_fun(I1,LS_im);
            eme_value_LS=EME_fun(LS_im(:,:,1),8);
            
            arr_SMPI=[arr_SMPI roundn(smpi_value,-4)];
            arr_EME=[arr_EME roundn(eme_value,-4)];
            
            arr_SMPI_gus=[arr_SMPI_gus roundn(smpi_value_gus,-4)];
            arr_EME_gus=[arr_EME_gus roundn(eme_value_gus,-4)];
            
            arr_SMPI_wave=[arr_SMPI_wave roundn(smpi_value_wave,-4)];
            arr_EME_wave=[arr_EME_wave roundn(eme_value_wave,-4)];
            
            arr_SMPI_dl=[arr_SMPI_dl roundn(smpi_value_dl,-4)];
            arr_EME_dl=[arr_EME_dl roundn(eme_value_dl,-4)];
            
            arr_SMPI_LS=[arr_SMPI_LS roundn(smpi_value_LS,-4)];
            arr_EME_LS=[arr_EME_LS roundn(eme_value_LS,-4)];
            %-----------------------是否保存图像，只算指标值时可以注释-----------------------------
            % imwrite(abs(I2) ,path_save_img);
            disp(name_class+'\'+name_class2+'\'+name_img)
        end
       
        mean1=mean(arr_SMPI);
        mean2=mean(arr_EME);
        mean1_gus=mean(arr_SMPI_gus);
        mean2_gus=mean(arr_EME_gus);
        mean1_wave=mean(arr_SMPI_wave);
        mean2_wave=mean(arr_EME_wave);
        mean1_dl=mean(arr_SMPI_dl);
        mean2_dl=mean(arr_EME_dl);
        mean1_LS=mean(arr_SMPI_LS);
        mean2_LS=mean(arr_EME_LS);
        
        arr2_SMPI=[arr2_SMPI;mean1];
         arr2_EME=[arr2_EME;mean2];
        
        arr2_SMPI_gus=[arr2_SMPI_gus;mean1_gus];
        arr2_EME_gus=[arr2_EME_gus;mean2_gus];
  
        arr2_SMPI_LS=[arr2_SMPI_LS;mean1_LS];
        arr2_EME_LS=[arr2_EME_LS;mean2_LS];
        
    end
    
    
    depression = angle_arr;
    class=sub_class_arr;
    SMPI=arr2_SMPI;
    EME=arr2_EME;
    SMPI_gus=arr2_SMPI_gus;
    EME_gus=arr2_EME_gus;
    SMPI_wave=arr2_SMPI_wave;
    EME_wave=arr2_EME_wave;
    SMPI_dl=arr2_SMPI_dl;
    EME_dl=arr2_EME_dl;
    SMPI_LS=arr2_SMPI_LS;
    EME_LS=arr2_EME_LS;
    
    T=table(depression,class,SMPI_gus,SMPI_wave,SMPI_dl,SMPI_LS,SMPI,EME_gus,EME_wave,EME_dl,EME_LS,EME);
    writetable(T,'july.xls')
    
end

disp('finish')
