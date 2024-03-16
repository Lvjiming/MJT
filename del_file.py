import os


def delete_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        os.remove(c_path)


def clear_file(path):
    filelist_father = os.listdir(path)
    for i in range(len(filelist_father)):
        filelist_son_path_qingkong = os.path.join(path, filelist_father[i])  # 输出一个路径
        delete_file(filelist_son_path_qingkong)
    print("清空原来delpic文件")
