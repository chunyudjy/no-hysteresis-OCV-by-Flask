import os
import glob

# dir = 'C:\\Users\\Aditya\\Desktop\\LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020\\0degC'

# files = glob.glob(os.path.join(dir, '*.mat'))

# for file_path in files:
#     os.remove(file_path)

# BASE_DIR = './'

dir_list = ['10degC', '25degC', '40degC', 'n10degC', 'n20degC']

for dir in dir_list:
    files_list = glob.glob(os.path.join(f'./{dir}', '*.mat'))
    for file_path in files_list:
        os.remove(file_path)
