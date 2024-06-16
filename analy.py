import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 파일 경로
image_files = ['num1.png', 'num2.png', 'num3.png']

# 주어진 RGB 값 (핑크색)
pink_rgb = [247, 205, 223]

# 이미지 파일 별로 처리
for idx, image_file in enumerate(image_files, start=1):
    print(f"Processing image {idx}: {image_file}")

    # 이미지 로드
    image = cv2.imread(image_file)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 주어진 RGB 값을 기반으로 핑크색 마스크 생성
    mask_pink = cv2.inRange(image, np.array(pink_rgb), np.array(pink_rgb))

    # 색상 범위 정의
    # 노란색
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])

    # 각 색상에 대한 마스크 생성
    mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)

    # 이미지를 5x5 그리드로 나누기
    num_rows = 5
    num_cols = 5
    cell_width = image.shape[1] // num_cols
    cell_height = image.shape[0] // num_rows

    # 각 그리드에 대한 통계 계산
    cell_stats_yellow = []  # 노랑 영역
    cell_stats_non_yellow = []  # 노랑이 아닌 영역
    for r in range(num_rows):
        for c in range(num_cols):
            # 현재 셀의 좌표 계산
            x1 = c * cell_width
            y1 = r * cell_height
            x2 = (c + 1) * cell_width
            y2 = (r + 1) * cell_height

            # 현재 셀의 영역과 노랑색, 핑크색 영역 계산
            cell_mask = np.zeros_like(mask_pink)
            cell_mask[y1:y2, x1:x2] = 255
            cell_yellow_area = np.count_nonzero(cv2.bitwise_and(cell_mask, mask_yellow))
            cell_non_yellow_area = np.count_nonzero(cv2.bitwise_and(cell_mask, ~mask_yellow))

            # 가중치 계산
            cell_total_area = (x2 - x1) * (y2 - y1)
            weight_yellow = cell_yellow_area / cell_total_area
            weight_non_yellow = cell_non_yellow_area / cell_total_area

            cell_stats_yellow.append(weight_yellow)
            cell_stats_non_yellow.append(weight_non_yellow)

    # 각 그리드의 노랑 영역 가중치 출력
    print("Yellow Area Weights in Grids:")
    for i, (weight_yellow, weight_non_yellow) in enumerate(zip(cell_stats_yellow, cell_stats_non_yellow), start=1):
        print(f"  Cell {i}: Yellow Weight = {weight_yellow:.2f}, Non-Yellow Weight = {weight_non_yellow:.2f}")

    # 그리드에 노랑 영역과 노랑이 아닌 영역의 가중치를 그리기
    grid_yellow = np.array(cell_stats_yellow).reshape((num_rows, num_cols))
    grid_non_yellow = np.array(cell_stats_non_yellow).reshape((num_rows, num_cols))

    # 그리드 시각화
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')

    cax1 = axes[0, 1].matshow(grid_yellow, cmap='YlGn')  # Choose your desired colormap
    axes[0, 1].set_title('Yellow Area Weights')
    fig.colorbar(cax1, ax=axes[0, 1])

    cax2 = axes[1, 0].matshow(grid_non_yellow, cmap='RdPu')  # Choose your desired colormap
    axes[1, 0].set_title('Non-Yellow Area Weights')
    fig.colorbar(cax2, ax=axes[1, 0])

    axes[1, 1].imshow(cv2.cvtColor(mask_pink, cv2.COLOR_GRAY2RGB))
    axes[1, 1].set_title('Pink Mask')

    # 그리드에 가중치 값 출력
    for i in range(num_rows):
        for j in range(num_cols):
            axes[0, 1].text(j, i, f'{grid_yellow[i, j]:.2f}', ha='center', va='center', color='black')
            axes[1, 0].text(j, i, f'{grid_non_yellow[i, j]:.2f}', ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()
