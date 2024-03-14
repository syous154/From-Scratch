# EfficientNet 구현

## 소개
본 레포지토리는 효율적인 모델 스케일링을 위해 설계된 EfficientNet의 구현을 담고 있습니다. EfficientNet 아키텍처는 다른 모델에 비해 훨씬 적은 파라미터와 FLOPs(Floating Point Operations)로 최신의 정확도를 달성합니다.

## 구현 세부사항
EfficientNet 모델 (B0-B7)은 PyTorch를 사용하여 `model.ipynb` 노트북 파일에 구현되어 있습니다. 각 모델은 원래 EfficientNet 논문에서 제안된 아키텍처를 따라 구성되었습니다.

## 사용법
본 레포지토리에서 구현된 EfficientNet 모델을 사용하려면, 간단히 `model.ipynb` 노트북 파일에서 해당 모델을 로드하고 원하는 작업에 사용하면 됩니다. 모델은 훈련이나 추론에 사용할 수 있습니다.

## 모델 파라미터
알파 (폭 스케일링 계수), 베타 (깊이 스케일링 계수), 감마 (해상도 스케일링 계수)와 같은 모델 파라미터는 EfficientNet 모델을 구성하는 데 중요합니다. 이러한 파라미터는 전체 모델 크기와 계산 비용을 결정합니다. 

이 구현 과정에서는 각 모델 변형에 대한 최적의 설정을 찾기 위해 이러한 파라미터에 대해 다양한 값들을 실험했습니다.

## 훈련과 평가
훈련을 진행할 시 `utility.ipynb` 파일 내에 있는 train_model 함수를 사용해도 되고 직접 만들어서 사용이 가능합니다.

평가 시에는 `parameters.ipynb` 파일 내에 있는 test_model 함수를 사용해도 되고 직접 만들어서 사용이 가능합니다.

## 결과
이 모델 EfficientNet 논문에 있는 모델의 파라미터 수를 비교해 보았습니다.


<img src="https://github.com/syous154/From-Scratch/assets/56266206/21bba1ca-3c29-4e29-ac31-573af7ddab11" width="400" height="400"/>
<img src="https://github.com/syous154/From-Scratch/assets/56266206/91200292-b1b8-4208-9364-91ee011e4db6" width="400" height="400"/>


파라미터 수를 보면 논문에 나온 값과 동일한 것을 알 수 있었습니다.


이 모델을 CIFAR-10 데이터 셋과 CIFAR-100 데이터 셋을 이용해 학습해보았습니다.

훈련 환경( epoch = 100, loss = crossEntrophy, optimizer = AdamW, lr = 0.001 )




EfficientNet-B1 (CIFAR-10)


![mineB1](https://github.com/syous154/From-Scratch/assets/56266206/51441cc2-dc69-434e-88fc-b815282b64a0)



EfficientNet-B3 (CIFAR-10)


![mineB3](https://github.com/syous154/From-Scratch/assets/56266206/4ca03a29-6bcb-4819-9e8d-129a91fc127d)



EfficinentNet-B5 (CIFAR-100)


<img width="681" alt="mineB5" src="https://github.com/syous154/From-Scratch/assets/56266206/a688d46e-93ea-4049-a7fb-87a5963bb501">


학습이 잘되고 모델의 크기 커질 수록 정확도도 증가하는 것을 볼 수 있었습니다.

개인적으로 torchvision 패키지 내에 있는 EfficientNet과 정확도를 비교해 보았을 때 거의 유사한 결과를 얻었습니다.
