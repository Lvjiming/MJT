import os
import matplotlib.pyplot as plt
import seaborn as sns

# 指定文件夹路径
folder_path = 'your path'

# 获取文件夹下所有图像文件
image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

for i in range(len(image_files)):
    # 读取第一张图像
    image_path = os.path.join(folder_path, image_files[i])
    sar_image = plt.imread(image_path)

    # 获取图像幅度
    amplitude = sar_image.flatten()*255

    plt.figure(figsize=(10, 5))
    # 显示原始SAR图像
    plt.subplot(1, 2, 1)
    plt.imshow(sar_image)
    plt.title('Original SAR Image')

    # 绘制图像幅度分布直方图和核密度估计曲线

    plt.subplot(1, 2, 2)
    sns.histplot(amplitude, bins=100, kde=True, color='blue', stat='probability',label='Amplitude Distribution')
    plt.title('Amplitude Distribution with Kernel Density Estimate')
    plt.legend()

    plt.tight_layout()
    plt.show()
