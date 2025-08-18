import os
import json
import glob

# JSON 파일들이 있는 디렉토리 경로 (WSL 환경을 가정)
directory_path = "/mnt/c/Users/Administrator/Desktop/synthrad2025-vardifformer/checkpoints/vdm_diffusion_separate_encoders_codebook-5"

# 디렉토리 내의 모든 .json 파일 목록을 가져옵니다.
json_files = glob.glob(os.path.join(directory_path, "*.json"))

# 계산된 값을 저장할 변수와 파일 경로를 초기화합니다.
min_calculated_value = float('inf')  # 값을 음의 무한대로 초기화하여 어떤 값과 비교해도 크게 만듦
best_file = None

# 각 JSON 파일을 순회하며 값을 계산합니다.
for file_path in json_files:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            # 필요한 모든 키가 JSON 데이터에 있는지 확인합니다.
            if all(key in data for key in ['fid', 'mse', 'ssim', 'psnr']):
                fid = data['fid']
                mse = data['mse']
                ssim = data['ssim']
                psnr = data['psnr']

                # fid + mse + ssim - psnr 값을 계산합니다.
                calculated_value = fid + mse + ssim - psnr
                
                # 현재 계산된 값이 max_calculated_value보다 크면 업데이트합니다.
                if calculated_value < min_calculated_value:
                    min_calculated_value = calculated_value
                    best_file = file_path
    
    except json.JSONDecodeError:
        # JSON 디코딩 오류가 발생한 파일은 건너뜁니다.
        continue
    except Exception as e:
        # 기타 예외가 발생한 파일은 건너뜁니다.
        continue

# 결과 출력
if best_file:
    print("가장 큰 'fid + mse + ssim - psnr' 값을 가진 파일:")
    print(f"파일 경로: {best_file}")
    print(f"계산된 값: {min_calculated_value}")
else:
    print("해당 디렉토리에서 필요한 모든 값을 가진 유효한 JSON 파일을 찾을 수 없습니다.")