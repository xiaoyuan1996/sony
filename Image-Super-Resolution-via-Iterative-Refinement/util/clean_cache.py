# **
# * Copyright @2022 AI, AIRCAS. (mails.ucas.ac.cn)
#
# @author yuanzhiqiang <yuanzhiqiang19@mails.ucas.ac.cn>
#         2022/06/28
import os, shutil

# 删除文件夹
def remove_dir(dir_path):
    try:
        shutil.rmtree(dir_path)
    except:
        print("file not exists: {}".format(dir_path))
        pass


def getFileFolderSize(fileOrFolderPath):
    """get size for file or folder"""
    totalSize = 0

    if not os.path.exists(fileOrFolderPath):
        return totalSize

    if os.path.isfile(fileOrFolderPath):
        totalSize = os.path.getsize(fileOrFolderPath)  # 5041481
        return totalSize

    if os.path.isdir(fileOrFolderPath):
        with os.scandir(fileOrFolderPath) as dirEntryList:
            for curSubEntry in dirEntryList:
                curSubEntryFullPath = os.path.join(fileOrFolderPath, curSubEntry.name)
                if curSubEntry.is_dir():
                    curSubFolderSize = getFileFolderSize(curSubEntryFullPath)  # 5800007
                    totalSize += curSubFolderSize
                elif curSubEntry.is_file():
                    curSubFileSize = os.path.getsize(curSubEntryFullPath)  # 1891
                    totalSize += curSubFileSize

            return totalSize


if __name__ == "__main__":
    path = "/data/diffussion/experiments/"
    for sub_path in os.listdir(path):
        check_path = os.path.join(path, sub_path)
        total = getFileFolderSize(check_path)
        try:
            visual_result = os.listdir(os.path.join(check_path, 'results'))
        except:
            remove_dir(check_path)
            continue

        if total < 40000 or len(visual_result) == 0:
            remove_dir(check_path)
        else:
            print("{} {}".format(check_path, total))