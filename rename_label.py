# 重命名图片文件名称

from glob import glob
import os
files = glob("./*/label.png")
i = 0
for file in files:
    i = i + 1
    print(file)
    name = file.split("\\")[-2].split("_")[-3] + "_" + file.split("\\")[-2].split("_")[-2]
    print(name)
    newname = name+".png"
    os.rename(file, newname)
    print(i)
