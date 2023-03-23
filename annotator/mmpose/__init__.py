import subprocess
import os
import cv2

def apply_mmpose(img):
    cv2.imwrite("tmp.png", img)
    mmpose_root = '/nvme/yangyifei/codes/xlab_sd/extensions/sd-webui-controlnet/annotator/mmpkg/modified_mmpose'
    
    line1 = "python"
    line1plus = os.path.join(mmpose_root, "demo/topdown_face_demo.py")
    line2 = os.path.join(mmpose_root, "configs/face_2d_keypoint/topdown_heatmap/300w/td-hm_hrnetv2-w18_8xb64-60e_300w-256x256.py")
    line3 = "https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_300w_256x256-eea53406_20211019.pth"
    line4 = "--input" 
    line5 = "tmp.png"
    line6 = "--output-root" 
    line7 = "tmp"
    
    subprocess.run([line1, line1plus, line2, line3, line4, line5, line6, line7], env=os.environ) 
    img_points = cv2.imread('tmp/tmp.png')
    return img_points
