% 指定文件夹路径
folderPath = 'D:\paper\mutilview\program\despeckle\SARimage';
folderPath_save = 'D:\paper\mutilview\program\despeckle\SARimage\hist';
% 获取文件夹下所有图像文件
imageFiles = dir(fullfile(folderPath, '*.png'));  % 假设图像格式为PNG，根据实际情况修改
FontSize=20;
%---------------------------------------------------------------------------
image_id=1;
imagePath = fullfile(folderPath, imageFiles(image_id).name);
sarImage = imread(imagePath);
% 获取图像幅度
amplitude = abs(sarImage(:));
% 显示原始SAR图像
imshow(sarImage);
title([imageFiles(image_id).name]);
% 绘制图像幅度分布直方图,'BinWidth' 是 histogram 函数的一个参数，用于指定直方图中每个条形箱的宽度
figure;
h = histogram(amplitude, 'Normalization', 'pdf', 'BinWidth', 2, 'EdgeColor', 'none');
h.FaceColor = [0.0 0.6 0.5];  % 设置柱状的颜色
title(sprintf('Coastal area \nRayleigh distribution'), 'FontSize', FontSize);
xlabel('Amplitude', 'FontSize', FontSize);  % 设置横坐标标签
ylabel('Probability Density', 'FontSize', FontSize);  % 设置纵坐标标签
grid on;  % 添加网格线
% 用ksdensity函数拟合数据
[f, xi] = ksdensity(amplitude, 'Kernel', 'triangle', 'Bandwidth', 5);
% 生成拟合曲线
hold on;
plot(xi, f, 'r', 'LineWidth', 2);
hold off;
% 保存直方图和拟合曲线
saveas(gcf, fullfile(folderPath_save, ['hist',imageFiles(image_id).name]));
close;
%---------------------------------------------------------------------------
image_id=2;
imagePath = fullfile(folderPath, imageFiles(image_id).name);
sarImage = imread(imagePath);
% 获取图像幅度
amplitude = abs(sarImage(:));
% 显示原始SAR图像
imshow(sarImage);
title([imageFiles(image_id).name]);
% 绘制图像幅度分布直方图,'BinWidth' 是 histogram 函数的一个参数，用于指定直方图中每个条形箱的宽度
figure;
h = histogram(amplitude, 'Normalization', 'pdf', 'BinWidth', 2, 'EdgeColor', 'none');
h.FaceColor = [0.0 0.6 0.5];  % 设置柱状的颜色
title(sprintf('Farmland area \n Nakagami distribution'), 'FontSize', FontSize);
xlabel('Amplitude', 'FontSize', FontSize);  % 设置横坐标标签
ylabel('Probability Density', 'FontSize', FontSize);  % 设置纵坐标标签
grid on;  % 添加网格线
% 用ksdensity函数拟合数据
[f, xi] = ksdensity(amplitude, 'Kernel', 'triangle', 'Bandwidth', 5);
% 生成拟合曲线
hold on;
plot(xi, f, 'r', 'LineWidth', 2);
hold off;
% 保存直方图和拟合曲线
saveas(gcf, fullfile(folderPath_save, ['hist',imageFiles(image_id).name]));
close;
%---------------------------------------------------------------------------
image_id=3;
imagePath = fullfile(folderPath, imageFiles(image_id).name);
sarImage = imread(imagePath);
% 获取图像幅度
amplitude = abs(sarImage(:));
% 显示原始SAR图像
imshow(sarImage);
title([imageFiles(image_id).name]);
% 绘制图像幅度分布直方图,'BinWidth' 是 histogram 函数的一个参数，用于指定直方图中每个条形箱的宽度
figure;
h = histogram(amplitude, 'Normalization', 'pdf', 'BinWidth', 2, 'EdgeColor', 'none');
h.FaceColor = [0.0 0.6 0.5];  % 设置柱状的颜色
title(sprintf('Forest \n K distribution'), 'FontSize', FontSize);
xlabel('Amplitude', 'FontSize', FontSize);  % 设置横坐标标签
ylabel('Probability Density', 'FontSize', FontSize);  % 设置纵坐标标签
grid on;  % 添加网格线
% 用ksdensity函数拟合数据
[f, xi] = ksdensity(amplitude, 'Kernel', 'triangle', 'Bandwidth', 5);
% 生成拟合曲线
hold on;
plot(xi, f, 'r', 'LineWidth', 2);
hold off;
% 保存直方图和拟合曲线
saveas(gcf, fullfile(folderPath_save, ['hist',imageFiles(image_id).name]));
close;
%---------------------------------------------------------------------------
image_id=4;
imagePath = fullfile(folderPath, imageFiles(image_id).name);
sarImage = imread(imagePath);
% 获取图像幅度
amplitude = abs(sarImage(:));
% 显示原始SAR图像
imshow(sarImage);
title([imageFiles(image_id).name]);
% 绘制图像幅度分布直方图,'BinWidth' 是 histogram 函数的一个参数，用于指定直方图中每个条形箱的宽度
figure;
h = histogram(amplitude, 'Normalization', 'pdf', 'BinWidth', 2, 'EdgeColor', 'none');
h.FaceColor = [0.0 0.6 0.5];  % 设置柱状的颜色
title(sprintf('City \n Log-normal distribution'), 'FontSize', FontSize);
xlabel('Amplitude', 'FontSize', FontSize);  % 设置横坐标标签
ylabel('Probability Density', 'FontSize', FontSize);  % 设置纵坐标标签
grid on;  % 添加网格线
% 用ksdensity函数拟合数据
[f, xi] = ksdensity(amplitude, 'Kernel', 'triangle', 'Bandwidth', 5);
% 生成拟合曲线
hold on;
plot(xi, f, 'r', 'LineWidth', 2);
hold off;
% 保存直方图和拟合曲线
saveas(gcf, fullfile(folderPath_save, ['hist',imageFiles(image_id).name]));
close;
%---------------------------------------------------------------------------
image_id=5;
imagePath = fullfile(folderPath, imageFiles(image_id).name);
sarImage = imread(imagePath);
% 获取图像幅度
amplitude = abs(sarImage(:));
% 显示原始SAR图像
imshow(sarImage);
title([imageFiles(image_id).name]);
% 绘制图像幅度分布直方图,'BinWidth' 是 histogram 函数的一个参数，用于指定直方图中每个条形箱的宽度
figure;
h = histogram(amplitude, 'Normalization', 'pdf', 'BinWidth', 2, 'EdgeColor', 'none');
h.FaceColor = [0.0 0.6 0.5];  % 设置柱状的颜色
title(sprintf('Mountain area\n Fisher distribution'),  'FontSize', FontSize);
xlabel('Amplitude', 'FontSize', FontSize);  % 设置横坐标标签
ylabel('Probability Density', 'FontSize', FontSize);  % 设置纵坐标标签
grid on;  % 添加网格线
% 用ksdensity函数拟合数据
[f, xi] = ksdensity(amplitude, 'Kernel', 'triangle', 'Bandwidth', 5);
% 生成拟合曲线
hold on;
plot(xi, f, 'r', 'LineWidth', 2);
hold off;
% 保存直方图和拟合曲线
saveas(gcf, fullfile(folderPath_save, ['hist',imageFiles(image_id).name]));
close;
%---------------------------------------------------------------------------
image_id=6;
imagePath = fullfile(folderPath, imageFiles(image_id).name);
sarImage = imread(imagePath);
% 获取图像幅度
amplitude = abs(sarImage(:));
% 显示原始SAR图像
imshow(sarImage);
title([imageFiles(image_id).name]);
% 绘制图像幅度分布直方图,'BinWidth' 是 histogram 函数的一个参数，用于指定直方图中每个条形箱的宽度
figure;
h = histogram(amplitude, 'Normalization', 'pdf', 'BinWidth', 2, 'EdgeColor', 'none');
h.FaceColor = [0.0 0.6 0.5];  % 设置柱状的颜色
title(sprintf('Sea surface \n Generalized gamma distribution'), 'FontSize', FontSize);
xlabel('Amplitude', 'FontSize', FontSize);  % 设置横坐标标签
ylabel('Probability Density', 'FontSize', FontSize);  % 设置纵坐标标签
grid on;  % 添加网格线
% 用ksdensity函数拟合数据
[f, xi] = ksdensity(amplitude, 'Kernel', 'triangle', 'Bandwidth', 5);
% 生成拟合曲线
hold on;
plot(xi, f, 'r', 'LineWidth', 2);
hold off;
% 保存直方图和拟合曲线
saveas(gcf, fullfile(folderPath_save, ['hist',imageFiles(image_id).name]));
close;

