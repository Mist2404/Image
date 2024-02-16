# 导入glob和os模块
import glob
import os
# 定义图片所在文件夹的路径，例如D:\images
images_path = "E:\py project\Imageclassification\dataset"
# 定义生成的txt文件的名字，例如label.txt
txt_name = "label.txt"
# 打开txt文件，以追加模式
f = open(txt_name, "a")
# 用glob.glob方法获取文件夹下所有jpg格式的图片的路径，返回一个列表
image_list = glob.glob(images_path + "\\*.jpg")
png = glob.glob(images_path + "\\*.png")
image_list = image_list + png
# 遍历这个列表，对于每个图片路径
for item in image_list:
    # 用os.path.basename方法获取图片的文件名，例如001.jpg
    img_name = os.path.basename(item)
    # 根据图片的文件名，给图片一个标签，这里假设图片的文件名的第一个字符就是标签，例如0
    label = img_name[0]
    # 将图片的路径和标签用制表符分隔，写入txt文件，换行
    f.write(item + "\t" + label + "\n")
# 关闭txt文件
f.close()
# 打印提示信息
print("生成txt成功！")
