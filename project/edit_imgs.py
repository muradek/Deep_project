import os

# change directory structure and images names to fit roboflow
# all images will be in the same directory with different names

directory = 'data'
for batch in os.listdir(directory):
    for img_name in os.listdir('data/' + batch):
        old_name = 'data/' + batch + "/" + img_name
        new_name = "new_imgs/" + batch + "_" + img_name
        os.rename(old_name, new_name)