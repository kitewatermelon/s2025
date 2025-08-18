import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

# 파일 경로
file1 = "/mnt/d/synthrad/Folder/0/sct_2ABA013.mha"
file2 = "/mnt/c/Users/Administrator/Desktop/synthrad2025-vardifformer/Val_Input/Task2/AB/2ABA013/cbct.mha"

# 이미지 불러오기
img1 = sitk.ReadImage(file1)
img2 = sitk.ReadImage(file2)

# numpy array로 변환
arr1 = sitk.GetArrayFromImage(img1)  # shape: (z, y, x)
arr2 = sitk.GetArrayFromImage(img2)

# 중앙 슬라이스 선택
mid1 = arr1.shape[0] // 2
mid2 = arr2.shape[0] // 2
print(f"Shape of arr1: {arr1.shape}, Shape of arr2: {arr2.shape}")
slice1 = arr1[mid1, :, :]
slice2 = arr2[mid2, :, :]

# 시각화
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(slice1, cmap="gray")
plt.title("sct_2ABA013.mha")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(slice2, cmap="gray")
plt.title("cbct.mha")
plt.axis("off")

plt.tight_layout()
plt.show()
