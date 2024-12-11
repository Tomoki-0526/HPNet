import os
import glob


data_dir = "/home/szj/HPNet/data/finetune/h5"
data_files = glob.glob(os.path.join(data_dir, "*.h5"))
data_num = len(data_files)


train_data = []
test_data = []
val_data = []


split_ratio = 0.7
split_index = int(split_ratio * data_num)


for i in range(split_index):
    train_data.append(str(i).zfill(3))
for j in range(split_index, data_num):
    test_data.append(str(j).zfill(3))
for k in range(1):
    val_data.append(str(k).zfill(3))


file = open(os.path.join(data_dir, "train_data.txt"), 'w')
for ele in train_data:
    file.write(str(ele).zfill(3) + "\n")
file.close()
file = open(os.path.join(data_dir, "test_data.txt"), 'w')
for ele in test_data:
    file.write(str(ele).zfill(3) + "\n")
file.close()
file = open(os.path.join(data_dir, "val_data.txt"), 'w')
for ele in val_data:
    file.write(str(ele).zfill(3) + "\n")
file.close()