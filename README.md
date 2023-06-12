# 캡스톤디자인2
## ColorSpaceConversionCompression
---

## 연구 배경
 초고화질 영상의 수요가 증가함에 따라 영상의 압축은 선택이 아닌 필수적이라 볼 수 있다. 이미 여러 분야에서 딥러닝을 접합고 있으며, 압축 역시 딥러닝을 활용하여 발전하고 있고, 다양한 학습 기반 압축 모델의 성능이 JPEG을 훨씬 앞서고 있다. 그러나 기존의 학습 기반 압축은 RGB 색공간에서 인코딩 / 디코딩을 실시하고 있기 때문에 사람의 시각적 특성을 이용하지 않고 있다. 또한 한 번의 학습으로 하나의 Point가 나오기 때문에 유연한 압축 수준을 결정할 수 없다는 문제점이 있다.

---
## 관련 연구
 우선 대표적인 학습 기반 압축 모델인 Minnen 모델이 있다. 이 모델은 이전 Balle 모델에서 Context 모델과 Entropy Parameters에 평균을 도입한 모델이다.
 
 다음으로는 DLEC 모델이 있다. 이 모델을 소개한 논문에서는 YCbCr 이미지를 입력으로 넣는 여러가지 방법을 제안하였는데, 그 중 YCbCr이미지를 한번에 입력으로 하는 Joint 모델을 사용하였다.
 
 마지막으로 Gained Unit이 있다. Gained Unit을 도입하면 이전 한 번의 학습에 하나의 Point가 나오던 것과는 달리 Gained 모델을 저장해서 생성된 Gained Unit간의 보간법을 통해 유연한 압축 퀄리티를 생성할 수 있게 된다.
 
 ---
 ## 연구 목표
  YCbCr이미지의 입력을 통해 인간의 시각적 특성을 이용함과 동시에 실제로 Y PSNR이 높아짐에 따라 주관적인 이미지 품질이 향상됨을 보인다.
  
  추가적으로 Gained Unit의 도입을 통해 한 번의 학습으로 여러가지 압축 퀄리티를 얻을 수 있도록 한다.
  
---
## 제안된 모델
![image](https://github.com/Sparkling-SAKE/ColorSpaceConversionCompression/assets/80191452/4236f93c-b176-468f-9226-9c761c4f279c)


# 원본과 압축 이미지 비교(Kodak24)
## Original
![image](https://github.com/Sparkling-SAKE/ColorSpaceConversionCompression/assets/80191452/a464ff7a-842d-4419-be95-8a8c9ce155b9)
## minnen
![image](https://github.com/Sparkling-SAKE/ColorSpaceConversionCompression/assets/80191452/a978b8e4-ad24-4437-8654-57c00e4381ee)
## minnen + DLEC
![image](https://github.com/Sparkling-SAKE/ColorSpaceConversionCompression/assets/80191452/14b9c354-418a-4611-98a7-4cde567e7d4f)
## minnen + DLEC + Bit Rate Control
![image](https://github.com/Sparkling-SAKE/ColorSpaceConversionCompression/assets/80191452/64664f25-29c4-4d34-b48d-7a76cd517d50)

# RD Curve
## - (PSNR_R + PSNR_G + PSNR_B) / 3
![image](https://github.com/Sparkling-SAKE/ColorSpaceConversionCompression/assets/80191452/ce4fb50e-2269-40c7-a270-50e7d8e427c6)
## - (4PSNR_Y + 1PSNR_Cb + 1PSNR_Cr) / 6
![image](https://github.com/Sparkling-SAKE/ColorSpaceConversionCompression/assets/80191452/2ec082dd-2ec1-4fe8-931f-f4b2904868ab)
## PSNR_Y
![image](https://github.com/Sparkling-SAKE/ColorSpaceConversionCompression/assets/80191452/5cb4f94e-2ebe-4a47-9558-c326425be1d7)


## 참고 문헌
Ballé, Johannes, et al, "Variational image compression with a scale hyperprior", 2018
Transform Network Architectures for Deep Learning Based End-to-End Image/Video Coding in Subsampled Color Spaces (IEEE Open Journal of signal processing, 2021)
D. Minnen, J. Balle, and G. D. Toderici, “Joint autoregressive and ´ hierarchical priors for learned image compression,” in Advances in Neural Information Processing Systems, vol. 31, 2018.
Z. Cui, J. Wang, S. Gao, T. Guo, Y. Feng, and B. Bai, “Asymmetric gained deep image compression with continuous rate adaptation,” in IEEE/CVF Comp. Vis. Patt. Recog. (CVPR), 2021.
O. Ugur Ulas, A. Murat Tekalp “Flexible luma-chroma bit allocation in learned image compression for high-fidelity sharper images” PCS, 2022.
