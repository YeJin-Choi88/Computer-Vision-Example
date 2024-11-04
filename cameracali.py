import cv2
import numpy as np

# 이미지 읽기(펴고 싶은 이미지)
image = cv2.imread('1.jpg')

# 카메라 캘리브레이션 매트릭스와 왜곡 계수 (예시 값, 사용자의 실제 캘리브레이션 값으로 변경 필요)
camera_matrix = np.array([[492.775068, 0.000000 , 739.116050],
                          [0.000000, 491.872567, 547.630382],
                          [0, 0, 1]])
dist_coeffs = np.array([-0.006288, 0.003272, -0.000889, 0.000885, 0.000000])  # 왜곡 계수

# 이미지 크기 가져오기
h, w = image.shape[:2]

# 보정된 이미지 맵 계산
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1)

# 왜곡 보정 적용
undistorted_image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

# 결과 저장 또는 표시
cv2.imwrite('undistorted_1.png', undistorted_image)
print("save finish")

#캘리 전 후 비교
combined_image = cv2.hconcat([image, undistorted_image]) 

# 결합된 이미지를 하나의 창에 표시
cv2.imshow('Combined Image', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
