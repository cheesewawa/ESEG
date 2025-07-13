import numpy as np
from scipy.sparse import coo_matrix
import cv2
from get_GTlabel_utils import *
from data.vis import visulize
import os
import tqdm
from datetime import datetime
from glob import glob
from multiprocessing import Process
#
# def generate_GroundTrueth_Edgemap(city_folder,edge_city_folder,vis_folder, edgemap_folder, city,r = 2):
#
#
#     labelimg_list = glob(os.path.join(city_folder,"*.png"))
#     insimg_list = glob(os.path.join(city_folder,"*.png"))
#     edgemap_list = glob(os.path.join(edgemap_folder, "*.png"))
#     labelimg_list.sort()
#     insimg_list.sort()
#     edgemap_list.sort()
#     print(labelimg_list, insimg_list)
#     for j in range(len(labelimg_list)):
#         name = labelimg_list[j].split('/')[-1][: ]
#         time = datetime.now().strftime("%c")
#         print("| time : %s | city : %s | [%i/%i] |"%(time,city,j,len(labelimg_list)))
#         # label_np = cv2.imread(labelimg_list[j],cv2.IMREAD_UNCHANGED)
#         # ins_np = cv2.imread(insimg_list[j],cv2.IMREAD_UNCHANGED)
#         # edgemap_np = cv2.imread(edgemap_list[j],cv2.IMREAD_UNCHANGED)
#
#         edgemap_np = np.array(Image.open(edgemap_list[j]).convert('L'))
#         label_np = np.array(Image.open(labelimg_list[j]))
#         ins_np = np.array(Image.open(insimg_list[j]))
#         # id --> trainid
#         # label_np = id2trainid(label_np)
#         # generate binary edge map
#         edge_binmap = edgemap_np#seg2edge(ins_np,r = r,label_ignore = [])
#         # initialize multiple semantic edge map
#         edge_semmap = np.zeros((NUM_TRAIN_CLASS,label_np.shape[0],label_np.shape[1]),dtype = np.bool_)
#         edge_semmap_splist = []
#         print(label_np)
#
#         for idx_cls in range(NUM_TRAIN_CLASS):
#             print(idx_cls)
#             idx_seg = label_np == idx_cls
#
#             if(idx_seg.sum() != 0):
#                 seg_map = np.zeros_like(label_np)
#                 seg_map[idx_seg] = ins_np[idx_seg]
#                 edge_semmap[idx_cls] = seg2edge_fast(seg_map,edge_binmap,2,[])
#                 print(edge_semmap[idx_cls])
#                 edge_semmap_splist.append(coo_matrix(edge_semmap[idx_cls]))
#             else:
#                 edge_semmap_splist.append(coo_matrix(np.zeros_like(label_np,dtype = np.bool_)))
#         edge_semmap_sparr = np.array(edge_semmap_splist)
#
#         # visualize generated multi-semantic edge map
#         edge_colormap = visulize(edge_semmap)
#
#         color_savepath = os.path.join("F:\lresizeandcrop640440\edge","%s_19edgecolor.png"%name)
#         cv2.imwrite(color_savepath,edge_colormap)
#
#
#         # change to sparse matrix for storeage effiency
#         # edge_semmap = coo_matrix(edge_semmap)
#         edge_savepath = os.path.join("F:\lresizeandcrop640440\edge","%s_19"%name)
#         np.savez_compressed(edge_savepath,edge_semmap_sparr)


import os
import numpy as np
from PIL import Image


def process_images(edge_dir, semantic_dir, output_dir, k):
    # 获取文件列表
    edge_files = sorted([f for f in os.listdir(edge_dir) if f.endswith('.png')])
    semantic_files = sorted([f for f in os.listdir(semantic_dir) if f.endswith('.png')])
    print(edge_files, semantic_files)

    # 确保边缘图和语义标签图数量相同
    assert len(edge_files) == len(semantic_files), "Edge and Semantic image counts do not match."

    for edge_file, semantic_file in zip(edge_files, semantic_files):
        print(edge_file)
        # 读取边缘图和语义标签图
        edge_image = np.array(Image.open(os.path.join(edge_dir, edge_file)).convert('L'))
        semantic_image = np.array(Image.open(os.path.join(semantic_dir, semantic_file)))

        # 获取图像尺寸
        height, width = edge_image.shape

        # 初始化11通道的语义边缘矩阵
        semantic_edge_matrix = np.zeros((11, height, width), dtype=bool)

        # 遍历边缘图上的每个像素
        for y in range(height):
            for x in range(width):
                if edge_image[y, x] != 0:  # 如果是边缘像素

                    # 计算正方形区域的边界
                    y_min = max(0, y - k // 2)
                    y_max = min(height, y + k // 2 + 1)
                    x_min = max(0, x - k // 2)
                    x_max = min(width, x + k // 2 + 1)

                    # 统计该区域内的语义信息
                    region = semantic_image[y_min:y_max, x_min:x_max]
                    unique_labels = np.unique(region)

                    # 更新语义边缘矩阵
                    for label in unique_labels:
                        if 0 <= label < 11:  # 确保语义标签在0-10范围内
                            semantic_edge_matrix[label, y, x] = True
                            # print(semantic_edge_matrix[label, y, x])


        edge_semmap = np.zeros((11, edge_image.shape[0], edge_image.shape[1]), dtype=np.bool_)
        edge_semmap_splist = []


        for idx_cls in range(11):

                edge_semmap_splist.append(coo_matrix(semantic_edge_matrix[idx_cls]))
                # print(semantic_edge_matrix[idx_cls])
                edge_semmap[idx_cls] = semantic_edge_matrix[idx_cls]

        edge_semmap_sparr = np.array(edge_semmap_splist)


        # visualize generated multi-semantic edge map
        edge_colormap = visulize(edge_semmap)

        color_savepath = os.path.join(output_dir,"%s_11edgecolor.png"%edge_file[:-4])
        cv2.imwrite(color_savepath,edge_colormap)


        # change to sparse matrix for storeage effiency
        # edge_semmap = coo_matrix(edge_semmap)
        edge_savepath = os.path.join(output_dir,"%s_11"%edge_file[:-4])
        np.savez_compressed(edge_savepath,edge_semmap_sparr)


        # print(semantic_edge_matrix)

        # 保存结果
        # output_path = os.path.join(output_dir, edge_file)
        # np.save(output_path, semantic_edge_matrix)


# 示例用法

edge_dir = r"F:\lresizeandcrop640440\resizedandcropped640440" #边缘图地址
semantic_dir = r"F:\SSH\semantic" #语义标签地址
output_dir = r'F:\lresizeandcrop640440\edge' #输出地址
k = 5  # 搜索正方形区域的边长

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 处理图像
process_images(edge_dir, semantic_dir, output_dir, k)



#
# # root_folder = "/home/martin/Documents/data/image_edge_dataset/cityscapes/data_orig"
# data_folder = r"F:\SSH\semantic" #os.path.join(root_folder,"gtFine","train")
# edge_folder = r"F:\lresizeandcrop640440\edge" #os.path.join(root_folder,"gtFine","train_edge")
# vis_folder = r"F:\lresizeandcrop640440\vis" #os.path.join(root_folder,"gtFine","train_edgevis")
# edgemap_folder = r"F:\lresizeandcrop640440\resizedandcropped640440"
#
# if os.path.exists(edge_folder) == False:
#     os.makedirs(edge_folder)
# if os.path.exists(vis_folder) == False:
#     os.makedirs(vis_folder)
# citys_list = os.listdir(data_folder)
# citys_list.sort()
# r = 2
#
# cityscapes_threads = []
# # for i in range(1):#range(len(citys_list)):
# #
# #     city_folder = os.path.join(data_folder)
# #     edge_city_folder = os.path.join(edge_folder)
# #     vis_city_folder = os.path.join(vis_folder)
# #     if os.path.exists(edge_city_folder) == False:
# #         os.makedirs(edge_city_folder)
# #     if os.path.exists(vis_city_folder) == False:
# #         os.makedirs(vis_city_folder)
#
# # generate_GroundTrueth_Edgemap(data_folder, edge_folder, vis_folder, edgemap_folder, "all", 2)
#
#

