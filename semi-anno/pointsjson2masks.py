# 根据通过labelme标注的点保存的json文件
# 通过sam-hq模型由点生成掩膜

import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import argparse
import glob
import os.path as osp
import numpy as np
import cv2
import base64
import json
import io
import PIL
import labelme
import matplotlib.pyplot as plt
from skimage.measure import approximate_polygon

from segment_anything import sam_model_registry, SamPredictor
import utils


parser = argparse.ArgumentParser(description="Road Marking Semi Annotations")
parser.add_argument("--json_dir", type=str, default=r"exp_data\point_jsons", help='The path of points json folder')
parser.add_argument("--output_dir", type=str, default=r"exp_data\images", help='The path of output json files folder') # 需要与要标注的图像在同一文件夹
parser.add_argument("--labels", type=str, default=r"exp_data\roadmarking_label.txt", help='The path of label files') # 标注文件，第一行为'_background_'
parser.add_argument("--vis_dir", type=str, default=r"exp_data\vis_results", help='The path of the folder wherer visualization results are saved') # 可视化结果保存路径
parser.add_argument("--annotation_dir", type=str, default=r"exp_data\annotations", help='The path of annotation masks folder') # label掩膜保存的文件夹路径
parser.add_argument("--ext", type=str, default=r".JPG", help='The suffix of the image to be labeled')

parser.add_argument("--model_type", type=str, default="vit_h", help='The model type of SAM-HQ') # 选择用于生成掩膜的SAM-HQ模型类型
parser.add_argument("--patch_size", type=int, default=480, help='The szie of patch image') # 以关键点为中心，从完整图像中切割patch size大小的图像

opt = parser.parse_args()


# 根据标注点获取图像块
# x轴为w方向 y轴为h方向
def points2patch(marking_points, img, patch_size):
    half_patch = patch_size // 2
    h, w, _ = img.shape

    # 点或者线
    cx, cy = np.mean(marking_points, axis=0)

    # 对边缘处进行判断
    if cx < half_patch:
        star_x = 0
    elif cx > (w - half_patch):
        star_x = w - patch_size
    else:
        star_x = cx - half_patch

    if cy < half_patch:
        star_y = 0
    elif cy > (h - half_patch):
        star_y = h - patch_size
    else:
        star_y = cy - half_patch

    star_y, star_x = int(star_y), int(star_x)
    patch_img = img[star_y : star_y + patch_size, star_x : star_x + patch_size, :]

    return patch_img, star_x, star_y

# 对生成的mask进行处理
def postprocess_mask(mask, star_x, star_y, patch_size, result_img, class_id):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w) * class_id
    result_img[star_y : star_y + patch_size, star_x : star_x + patch_size] = mask_image

    vis_mask = np.zeros_like(result_img) 
    vis_mask[star_y : star_y + patch_size, star_x : star_x + patch_size] = mask

    return result_img, vis_mask


# 展示标注的点与获取掩膜结果
def show_mask(mask, ax, seed):
    # 由类别来设置种子
    np.random.seed(seed)
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=50):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 

def show_res(masks, class_list, input_points, input_labels, name, image, save_dir):
    mydpi = 72
    h, w, _ = image.shape
    plt.figure(figsize=(w / mydpi,h / mydpi) ,dpi=mydpi)
    plt.imshow(image)

    for mask, class_id in zip(masks, class_list):
        show_mask(mask, plt.gca(), class_id)

    show_points(input_points, input_labels, plt.gca())
        
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, name + '.png'), bbox_inches='tight',pad_inches=-0.1, dpi=mydpi)
    plt.close()


def pointsjson2masks(input_dir, labels, patch_size, predictor, vis_dir, annotation_dir):
    class_name_to_id = {}
    for class_id, line in enumerate(open(labels).readlines()):
        # _background_ 为背景，类别id为0
        class_name = line.strip()
        class_name_to_id[class_name] = class_id

    masks_dict = {} # label掩膜

    for filename in glob.glob(osp.join(input_dir, "*.json")):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)
        img = labelme.utils.img_data_to_arr(label_file.imageData) # 读取完整图像

        result_label = np.zeros([img.shape[0], img.shape[1]]).astype(np.int32) # 为了对生成label掩膜

        # 用于对半自动标注的结果可视化
        vis_masks = [] 
        class_list = []
        input_point_all = []

        # 遍历标注的点
        for shape in label_file.shapes:
            # 只支持点或者线点的方式
            if (shape["shape_type"] != "point") & (shape["shape_type"] != "line") &  (shape["shape_type"] != "linestrip"):
                print(
                    "Skipping shape: label={label}, "
                    "shape_type={shape_type}".format(**shape)
                )
                continue

            marking_points = np.array(shape["points"])
            patch_img, star_x, star_y = points2patch(marking_points, img, patch_size)

            input_point = marking_points - np.array([star_x, star_y])
            input_label = np.ones(input_point.shape[0])

            input_box = None
            hq_token_only = False

            predictor.set_image(patch_img)

            masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box = input_box,
                    multimask_output=False,
                    hq_token_only=hq_token_only, 
                )
                        
            class_name = shape["label"]
            class_id = class_name_to_id[class_name]
       
            class_list.append(class_id)
            input_point_all.append(marking_points)

            result_label, vis_mask = postprocess_mask(masks[0], star_x, star_y, patch_size, result_label, class_id)
            vis_masks.append(vis_mask)

        input_points = np.vstack(input_point_all)
        input_labels = np.ones(input_points.shape[0])
        name, _  = osp.splitext(osp.basename(filename))
        show_res(vis_masks, class_list, input_points, input_labels, name, img, vis_dir)
        cv2.imwrite(os.path.join(annotation_dir, name + '.png'), result_label)
        masks_dict[name] = result_label

    return masks_dict, class_name_to_id

# 通过掩膜中的多边形生成labelme的json文件中shape字段
def get_polygon_shape(mask, class_name_to_id):
    # 排除_background_这一类别
    class_list = list(range(len(class_name_to_id)))[1:]
    shape = []
    for value in class_list:
        img = np.zeros_like(mask)
        img[mask == value] = 1
        if np.sum(img) == 0:
            continue
        contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        label = [k for k, v in class_name_to_id.items() if v == value][0]
        for c in contours:
            poly = {}
            poly["label"] = label
            points = c.reshape([-1, 2]).astype(np.float32)

            arr_points = approximate_polygon(points, tolerance=1)

            poly["points"] = arr_points.tolist() 
            poly["group_id"] = None
            poly["shape_type"] = "polygon"
            poly["flag"] = {}

            shape.append(poly)

    return shape

# 获取图像宽高
def get_image_height_and_width(imageData):
    img_arr = utils.img_b64_to_arr(imageData)
    imageHeight = img_arr.shape[0]
    imageWidth = img_arr.shape[1]
    return imageHeight, imageWidth

# 加载图像
def load_image_file(filename):
    image_pil = PIL.Image.open(filename)
    # apply orientation to image according to exif
    image_pil = utils.apply_exif_orientation(image_pil)

    with io.BytesIO() as f:
        ext = os.path.splitext(filename)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            format = "JPEG"
        else:
            format = "PNG"
        image_pil.save(f, format=format)
        f.seek(0)

        return f.read()            

# 根据掩膜生成labelme格式的json文件
def masks2json(masks_dict, class_name_to_id, ext, output_dir):

    for name, mask in masks_dict.items():       
        imagePath = name + ext
        imageData = load_image_file(os.path.join(output_dir, imagePath))
        imageData = base64.b64encode(imageData).decode("utf-8")
        imageHeight, imageWidth = get_image_height_and_width(imageData)

        shape = get_polygon_shape(mask, class_name_to_id)
        data = dict(
            version='5.1.1',
            flags={},
            shapes=shape,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,)
        
        with open(osp.join(output_dir, name+'.json'), "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print('Save json file at: {}'.format(osp.join(output_dir, name+'.json')))      

def main(opt):
    json_dir = opt.json_dir
    output_dir = opt.output_dir
    labels = opt.labels
    vis_dir = opt.vis_dir
    annotation_dir = opt.annotation_dir
    ext = opt.ext

    patch_size = opt.patch_size
    model_type = opt.model_type

    sam_checkpoint = "./pretrained_checkpoint/sam_hq_{}.pth".format(model_type)
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 获取SAM-HQ生成的掩膜，保存可视化效果图与label掩膜
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)
    masks_dict, class_name_to_id = pointsjson2masks(json_dir, labels, patch_size, predictor, vis_dir, annotation_dir) 

    # 根据label掩膜生成labelme格式的json文件，方便后续的人工检查与修改
    masks2json(masks_dict, class_name_to_id, ext, output_dir)


if __name__ == "__main__":
    main(opt)