import os


def re_name(root):
    filelist_father = os.listdir(root)

    print(filelist_father)

    for i in range(len(filelist_father)):
        filelist_son_path = os.path.join(root, filelist_father[i])  # Output a path
        filelist_son = os.listdir(filelist_son_path)  # Output a list of all tags
        i += 1
        n = 0
        for j in filelist_son:
            oldname = filelist_son_path + os.sep + filelist_son[n]  # Output a path
            newname = filelist_son_path + os.sep + str(n) + '.png'
            os.rename(oldname, newname)
            print(oldname, newname)
            n += 1


def re_name_2(root):
    filelist_father = os.listdir(root)
    print(filelist_father)

    for i in range(len(filelist_father)):
        filelist_son_path = os.path.join(root, filelist_father[i])  # 输出一个路径
        filelist_son = os.listdir(filelist_son_path)  # 输出一个包含所有标签的list
        i += 1
        n = 0

        for j in filelist_son:
            oldname = filelist_son_path + os.sep + filelist_son[n]
            newname = filelist_son_path + os.sep + j.split("_", 5)[-1]
            if os.path.exists(newname):
                print('名字重复')
                os.remove(oldname)
            else:
                os.rename(oldname, newname)

            print(oldname, newname)
            n += 1


def re_name_3(root):
    filelist_father = os.listdir(root)
    print(filelist_father)

    for i in range(len(filelist_father)):
        filelist_son_path = os.path.join(root, filelist_father[i])
        filelist_son = os.listdir(filelist_son_path)
        i += 1
        n = 0

        for j in filelist_son:
            oldname = filelist_son_path + os.sep + filelist_son[n]
            newname = filelist_son_path + os.sep + j.split("_", 5)[-2] + j.split("_", 5)[-1]
            if os.path.exists(newname):
                print('名字重复')
                os.remove(oldname)
            else:
                os.rename(oldname, newname)
            print(oldname, newname)
            n += 1
