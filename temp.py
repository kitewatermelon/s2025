import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def visualize_middle_slices(mha_path, num_slices=5):
    """
    가운데 slice 기준으로 앞뒤 포함하여 총 num_slices 개의 CT 슬라이스 시각화
    """
    ct_image = sitk.ReadImage(mha_path)
    ct_array = sitk.GetArrayFromImage(ct_image)  # (slice, height, width)
    print("Shape (slice, height, width):", ct_array.shape)

    middle_idx = ct_array.shape[0] // 2
    half = num_slices // 2
    indices = [middle_idx + i for i in range(-half, half + 1)]

    fig, axes = plt.subplots(1, num_slices, figsize=(3 * num_slices, 5))
    for ax, idx in zip(axes, indices):
        if 0 <= idx < ct_array.shape[0]:
            ax.imshow(ct_array[idx], cmap='gray')
            ax.set_title(f"Slice {idx}")
            ax.axis('off')
        else:
            ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def rename_cbct_to_sct(directory):
    """
    디렉토리 내 'cbct_'로 시작하는 .mha 파일을 'sct_'로 시작하도록 이름 변경
    """
    for filename in os.listdir(directory):
        if filename.endswith(".mha") and "_" in filename:
            parts = filename.split("_", 1)
            new_filename = "sct_" + parts[1]
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"[RENAME] '{filename}' -> '{new_filename}' 완료")
    print("📦 파일 이름 변경 완료.")


def transpose_and_save_mha(directory):
    """
    (H, W, Slice) 형식의 .mha 파일을 (Slice, H, W) 형식으로 바꾸어 저장
    """
    for filename in os.listdir(directory):
        if filename.startswith("sct_") and filename.endswith(".mha"):
            filepath = os.path.join(directory, filename)
            image = sitk.ReadImage(filepath)
            array = sitk.GetArrayFromImage(image)

            if array.ndim != 3:
                print(f"[SKIP] {filename}: 예상과 다른 차원 {array.shape}")
                continue

            # (H, W, Slice) → (Slice, H, W)
            transposed = np.transpose(array, (2, 0, 1))

            new_image = sitk.GetImageFromArray(transposed)
            new_image.SetSpacing(image.GetSpacing())
            new_image.SetOrigin(image.GetOrigin())
            new_image.SetDirection(image.GetDirection())

            sitk.WriteImage(new_image, filepath)
            print(f"[SAVE] '{filename}' saved with shape {transposed.shape}")

    print("✅ 모든 파일 변환 및 저장 완료.")


def main():
    # 수정 경로
    vis_path = "/mnt/d/synthrad/Folder/0/0_2HND053.mha"
    # vis_path = "./synthRAD2025_Task2_Val_Input_D/Task2/HN/2HND053/cbct.mha"
    data_dir = "/mnt/d/synthrad/Folder/0"
    # print(vis_path)
    # # 1. 슬라이스 시각화
    # visualize_middle_slices(vis_path)
    rename_cbct_to_sct(data_dir)
    # transpose_and_save_mha(data_dir)

if __name__ == "__main__":
    main()


# # 원본 디렉토리 경로
# directory = "/mnt/d/synthrad/Folder/0"

# # 모든 파일 이름과 매핑된 이전 이름들 기록
# name_map = {
#     'sct_A013':'0_2ABA013' ,
#     'sct_A015':'0_2ABA015' ,
#     'sct_A017':'0_2ABA017' ,
#     'sct_A059':'0_2ABA059' ,
#     'sct_A061':'0_2ABA061' ,
#     'sct_A062':'0_2ABA062' ,
#     'sct_A092':'0_2ABA092' ,
#     'sct_A107':'0_2ABA107' ,
#     'sct_A121':'0_2ABA121' ,
#     'sct_A127':'0_2ABA127' ,
#     'sct_B003':'0_2ABB003' ,
#     'sct_B008':'0_2ABB008' ,
#     'sct_B017':'0_2ABB017' ,
#     'sct_B035':'0_2ABB035' ,
#     'sct_B046':'0_2ABB046' ,
#     'sct_B058':'0_2ABB058' ,
#     'sct_B062':'0_2ABB062' ,
#     'sct_B083':'0_2ABB083' ,
#     'sct_B099':'0_2ABB099' ,
#     'sct_B107':'0_2ABB107' ,
#     'sct_C013':'0_2ABC013' ,
#     'sct_C033':'0_2ABC033' ,
#     'sct_C062':'0_2ABC062' ,
#     'sct_C069':'0_2ABC069' ,
#     'sct_C091':'0_2ABC091' ,
#     'sct_C111':'0_2ABC111' ,
#     'sct_C149':'0_2ABC149' ,
#     'sct_C202':'0_2ABC202' ,
#     'sct_C217':'0_2ABC217' ,
#     'sct_C221':'0_2ABC221' ,
#     'sct_D010':'0_2ABD010' ,
#     'sct_D016':'0_2ABD016' ,
#     'sct_D039':'0_2ABD039' ,
#     'sct_D045':'0_2ABD045' ,
#     'sct_D057':'0_2ABD057' ,
#     'sct_D076':'0_2ABD076' ,
#     'sct_D090':'0_2ABD090' ,
#     'sct_D107':'0_2ABD107' ,
#     'sct_E112':'0_2ABE112' ,
#     'sct_E113':'0_2ABE113' ,
#     'sct_E123':'0_2ABE123' ,
#     'sct_E132':'0_2ABE132' ,
#     'sct_E134':'0_2ABE134' ,
#     'sct_E143':'0_2ABE143' ,
#     'sct_E157':'0_2ABE157' ,
#     'sct_E168':'0_2ABE168' ,
#     'sct_E171':'0_2ABE171' ,
#     'sct_E179':'0_2ABE179' ,
#     'sct_A004':'0_2HNA004' ,
#     'sct_A006':'0_2HNA006' ,
#     'sct_A012':'0_2HNA012' ,
#     'sct_A042':'0_2HNA042' ,
#     'sct_A049':'0_2HNA049' ,
#     'sct_A053':'0_2HNA053' ,
#     'sct_A054':'0_2HNA054' ,
#     'sct_A063':'0_2HNA063' ,
#     'sct_A066':'0_2HNA066' ,
#     'sct_A069':'0_2HNA069' ,
#     'sct_B004':'0_2HNB004' ,
#     'sct_B018':'0_2HNB018' ,
#     'sct_B029':'0_2HNB029' ,
#     'sct_B042':'0_2HNB042' ,
#     'sct_B049':'0_2HNB049' ,
#     'sct_B056':'0_2HNB056' ,
#     'sct_B066':'0_2HNB066' ,
#     'sct_B073':'0_2HNB073' ,
#     'sct_B088':'0_2HNB088' ,
#     'sct_B092':'0_2HNB092' ,
#     'sct_C019':'0_2HNC019' ,
#     'sct_C022':'0_2HNC022' ,
#     'sct_C054':'0_2HNC054' ,
#     'sct_C064':'0_2HNC064' ,
#     'sct_C065':'0_2HNC065' ,
#     'sct_C083':'0_2HNC083' ,
#     'sct_C096':'0_2HNC096' ,
#     'sct_C105':'0_2HNC105' ,
#     'sct_C107':'0_2HNC107' ,
#     'sct_C111':'0_2HNC111' ,
#     'sct_D007':'0_2HND007' ,
#     'sct_D026':'0_2HND026' ,
#     'sct_D029':'0_2HND029' ,
#     'sct_D030':'0_2HND030' ,
#     'sct_D039':'0_2HND039' ,
#     'sct_D040':'0_2HND040' ,
#     'sct_D052':'0_2HND052' ,
#     'sct_D053':'0_2HND053' ,
#     'sct_D064':'0_2HND064' ,
#     'sct_D074':'0_2HND074' ,
#     'sct_E002':'0_2HNE002' ,
#     'sct_E008':'0_2HNE008' ,
#     'sct_E013':'0_2HNE013' ,
#     'sct_E027':'0_2HNE027' ,
#     'sct_E035':'0_2HNE035' ,
#     'sct_E044':'0_2HNE044' ,
#     'sct_E045':'0_2HNE045' ,
#     'sct_E048':'0_2HNE048' ,
#     'sct_E062':'0_2HNE062' ,
#     'sct_E071':'0_2HNE071' ,
#     'sct_A003':'0_2THA003' ,
#     'sct_A007':'0_2THA007' ,
#     'sct_A008':'0_2THA008' ,
#     'sct_A011':'0_2THA011' ,
#     'sct_A014':'0_2THA014' ,
#     'sct_A024':'0_2THA024' ,
#     'sct_A026':'0_2THA026' ,
#     'sct_A027':'0_2THA027' ,
#     'sct_A044':'0_2THA044' ,
#     'sct_A089':'0_2THA089' ,
#     'sct_B005':'0_2THB005' ,
#     'sct_B016':'0_2THB016' ,
#     'sct_B018':'0_2THB018' ,
#     'sct_B032':'0_2THB032' ,
#     'sct_B048':'0_2THB048' ,
#     'sct_B056':'0_2THB056' ,
#     'sct_B066':'0_2THB066' ,
#     'sct_B084':'0_2THB084' ,
#     'sct_B093':'0_2THB093' ,
#     'sct_B107':'0_2THB107' ,
#     'sct_C004':'0_2THC004' ,
#     'sct_C014':'0_2THC014' ,
#     'sct_C028':'0_2THC028' ,
#     'sct_C041':'0_2THC041' ,
#     'sct_C046':'0_2THC046' ,
#     'sct_C051':'0_2THC051' ,
#     'sct_C054':'0_2THC054' ,
#     'sct_C071':'0_2THC071' ,
#     'sct_C075':'0_2THC075' ,
#     'sct_C121':'0_2THC121' ,
#     'sct_D032':'0_2THD032' ,
#     'sct_D033':'0_2THD033' ,
#     'sct_D046':'0_2THD046' ,
#     'sct_D051':'0_2THD051' ,
#     'sct_D057':'0_2THD057' ,
#     'sct_D058':'0_2THD058' ,
#     'sct_D065':'0_2THD065' ,
#     'sct_D066':'0_2THD066' ,
#     'sct_D072':'0_2THD072' ,
#     'sct_D077':'0_2THD077' ,
#     'sct_E005':'0_2THE005' ,
#     'sct_E016':'0_2THE016' ,
#     'sct_E029':'0_2THE029' ,
#     'sct_E043':'0_2THE043' ,
#     'sct_E054':'0_2THE054' ,
#     'sct_E074':'0_2THE074' ,
#     'sct_E078':'0_2THE078' ,
#     'sct_E082':'0_2THE082' ,

# }

# # 디렉토리 내 파일 이름 순회
# for filename in os.listdir(directory):
#     if filename.endswith(".mha"):
#         name_without_ext = filename[:-4]  # 확장자 제거
#         if name_without_ext in name_map:
#             new_name = name_map[name_without_ext] + ".mha"
#             old_path = os.path.join(directory, filename)
#             new_path = os.path.join(directory, new_name)
#             os.rename(old_path, new_path)
#             print(f"[RENAME] '{filename}' -> '{new_name}' 완료")

# print("모든 파일 이름 복원 완료.")
