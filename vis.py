# app.py
import streamlit as st
import SimpleITK as sitk
import numpy as np
from PIL import Image

# 1. MHA 파일 로드
file_name = "2ABA013"
cbct_path = f"./Val_Input/Task2/AB/{file_name}/cbct.mha"
ct_path   = f"/mnt/d/synthrad/Folder/0/sct_{file_name}.mha"

cbct_vol = sitk.ReadImage(cbct_path)
ct_vol   = sitk.ReadImage(ct_path)

cbct_array = sitk.GetArrayFromImage(cbct_vol).astype(np.float32)  # (D, H, W)
ct_array   = sitk.GetArrayFromImage(ct_vol).astype(np.float32)

# 2. Streamlit UI
st.title("CBCT vs CT Viewer with Absolute Error")

num_slices = cbct_array.shape[0]
slice_idx = st.slider("Select slice", 0, num_slices - 1, 0)

# 선택한 슬라이스
cbct_slice = cbct_array[slice_idx, :, :]
ct_slice   = ct_array[slice_idx, :, :]

# 이미지 정규화 함수
def normalize_img(img):
    img = img - np.min(img)
    if np.max(img) > 0:
        img = img / np.max(img)
    img = (img * 255).astype(np.uint8)
    return img

cbct_img = Image.fromarray(normalize_img(cbct_slice))
ct_img   = Image.fromarray(normalize_img(ct_slice))

# 절대 오차 계산
abs_error_slice = np.abs(cbct_slice - ct_slice)
abs_error_img = Image.fromarray(normalize_img(abs_error_slice))

# 3개 이미지를 동시에 보여주기
col1, col2, col3 = st.columns(3)
col1.image(cbct_img, caption=f"CBCT Slice {slice_idx}", use_column_width=True)
col2.image(ct_img, caption=f"CT Slice {slice_idx}", use_column_width=True)
col3.image(abs_error_img, caption=f"Absolute Error Slice {slice_idx}", use_column_width=True)
