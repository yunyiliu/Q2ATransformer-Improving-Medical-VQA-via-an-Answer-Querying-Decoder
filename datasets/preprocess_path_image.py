
import os
import shutil
import json

# Image data

# imgid2id = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/imgid2idx.json','r'))
# part1_img = []
# path_part1 = '/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/images/part1'
# path_part2 = '/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/images/part2'
# target_path = '/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/images'
# part2_img = []
# for img in os.listdir(path_part1):
#     if "jpg" in img:
#         shutil.copy(os.path.join(path_part1,img), os.path.join(target_path, 'image'))
# for f in os.listdir(path_part2):
#     cur_path = path_part2 + '/' + f
#     for img in os.listdir(cur_path):
#         if "jpg" in img:
#             shutil.copy(os.path.join(cur_path, img), os.path.join(target_path, 'image'))

# final_path = '/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/images/image'
# i = 0
# for img in os.listdir(final_path):
#     print(i)
#     src = os.path.join(os.path.abspath(final_path), img)
#     dst = os.path.join(os.path.abspath(final_path), str(i)+ '.jpg')
#     os.rename(src, dst)
#     i = i + 1

# print(len(os.listdir(final_path)))

# step1: 把image传过来，放到path_VQA下面ssh -p 38434 root@region-3.autodl.com
# step2： 把所有image copy到一个path_VQA/images 的image folder下面，需要有5003张图片,但是只有4883张
# target_path = '/root/VQA_Main/Modified_MedVQA-main/data/path_VQA'
# test_img_path = '/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/test'
# train_img_path_t0 = '/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/t0'
# train_img_path_t1 = '/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/t1'
# train_img_path_t2 = '/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/t2'
# train_img_path_t3 = '/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/t3'
# train_img_path_t4 = '/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/t4'
# train_img_path_t5 = '/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/t5'

# val_img_path = '/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/val'
# test_count = 0
# for f_test in os.listdir(test_img_path):
#     cur_path = os.path.join(test_img_path, f_test)
#     for img in os.listdir(cur_path):
#         if "jpg" in img:
#             shutil.copy(os.path.join(cur_path, img), os.path.join(target_path, 'images'))

# for f_val in os.listdir(val_img_path):
#     cur_path = os.path.join(val_img_path, f_val)
#     for img in os.listdir(cur_path):
#         if "jpg" in img:
#             shutil.copy(os.path.join(cur_path, img), os.path.join(target_path, 'images'))
            
# for f1 in os.listdir(train_img_path_t0):
#     file_path = os.path.join(train_img_path_t0, f1)
#     if(f1 == 'train'): 
#         for f_train in os.listdir(file_path):
#             cur_path = os.path.join(file_path, f_train)
#             for img in os.listdir(cur_path):
#                 if "jpg" in img:
#                     shutil.copy(os.path.join(cur_path, img), os.path.join(target_path, 'images'))
#     else: 
#         for img in os.listdir(file_path):
#             if "jpg" in img:
#                 shutil.copy(os.path.join(file_path, img), os.path.join(target_path, 'images'))       

# for f1 in os.listdir(train_img_path_t1):
#     file_path = os.path.join(train_img_path_t1, f1)
#     if(f1 == 'train'): 
#         for f_train in os.listdir(file_path):
#             cur_path = os.path.join(file_path, f_train)
#             for img in os.listdir(cur_path):
#                 if "jpg" in img:
#                     shutil.copy(os.path.join(cur_path, img), os.path.join(target_path, 'images'))
#     else: 
#         for img in os.listdir(file_path):
#             if "jpg" in img:
#                 shutil.copy(os.path.join(file_path, img), os.path.join(target_path, 'images'))       

# for f1 in os.listdir(train_img_path_t2):
#     file_path = os.path.join(train_img_path_t2, f1)
#     if(f1 == 'train'): 
#         for f_train in os.listdir(file_path):
#             cur_path = os.path.join(file_path, f_train)
#             for img in os.listdir(cur_path):
#                 if "jpg" in img:
#                     shutil.copy(os.path.join(cur_path, img), os.path.join(target_path, 'images'))
#     else: 
#         for img in os.listdir(file_path):
#             if "jpg" in img:
#                 shutil.copy(os.path.join(file_path, img), os.path.join(target_path, 'images'))       

# for f1 in os.listdir(train_img_path_t3):
#     file_path = os.path.join(train_img_path_t3, f1)
#     if(f1 == 'train'): 
#         for f_train in os.listdir(file_path):
#             cur_path = os.path.join(file_path, f_train)
#             for img in os.listdir(cur_path):
#                 if "jpg" in img:
#                     shutil.copy(os.path.join(cur_path, img), os.path.join(target_path, 'images'))
#     else: 
#         for img in os.listdir(file_path):
#             if "jpg" in img:
#                 shutil.copy(os.path.join(file_path, img), os.path.join(target_path, 'images'))       

# for f1 in os.listdir(train_img_path_t4):
#     file_path = os.path.join(train_img_path_t4, f1)
#     if(f1 == 'train'): 
#         for f_train in os.listdir(file_path):
#             cur_path = os.path.join(file_path, f_train)
#             for img in os.listdir(cur_path):
#                 if "jpg" in img:
#                     shutil.copy(os.path.join(cur_path, img), os.path.join(target_path, 'images'))
#     else: 
#         for img in os.listdir(file_path):
#             if "jpg" in img:
#                 shutil.copy(os.path.join(file_path, img), os.path.join(target_path, 'images'))       
                
# for f1 in os.listdir(train_img_path_t5):
#     file_path = os.path.join(train_img_path_t5, f1)
#     if(f1 == 'train'): 
#         for f_train in os.listdir(file_path):
#             cur_path = os.path.join(file_path, f_train)
#             for img in os.listdir(cur_path):
#                 if "jpg" in img:
#                     shutil.copy(os.path.join(cur_path, img), os.path.join(target_path, 'images'))
#     else: 
#         for img in os.listdir(file_path):
#             if "jpg" in img:
#                 shutil.copy(os.path.join(file_path, img), os.path.join(target_path, 'images'))       
                
# all_img = os.listdir('/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/images')
# print(len(all_img))

target_path = '/root/VQA_Main/Modified_MedVQA-main/data/path_VQA'
test_img_path = '/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/test'
train_img_path = '/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/train'
val_img_path = '/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/val'

for img in os.listdir(test_img_path):
    if "jpg" in img:
        shutil.copy(os.path.join(test_img_path, img), os.path.join(target_path, 'images'))
 
for img in os.listdir(train_img_path):
    if "jpg" in img:
        shutil.copy(os.path.join(train_img_path, img), os.path.join(target_path, 'images'))
 
for img in os.listdir(val_img_path):
    if "jpg" in img:
        shutil.copy(os.path.join(val_img_path, img), os.path.join(target_path, 'images'))
  
            
        
all_img = os.listdir('/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/images')
print(len(all_img))