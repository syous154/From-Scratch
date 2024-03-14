# EfficientNet 구현

## 소개
본 레포지토리는 효율적인 모델 스케일링을 위해 설계된 EfficientNet의 구현을 담고 있습니다. EfficientNet 아키텍처는 다른 모델에 비해 훨씬 적은 파라미터와 FLOPs(Floating Point Operations)로 최신의 정확도를 달성합니다.

## 구현 세부사항
EfficientNet 모델 (B0-B7)은 PyTorch를 사용하여 `model.ipynb` 노트북 파일에 구현되어 있습니다. 각 모델은 원래 EfficientNet 논문에서 제안된 아키텍처를 따라 구성되었습니다.

## 사용법
본 레포지토리에서 구현된 EfficientNet 모델을 사용하려면, 간단히 `model.ipynb` 노트북 파일에서 해당 모델을 로드하고 원하는 작업에 사용하면 됩니다. 모델은 훈련이나 추론에 사용할 수 있습니다.

## 모델 파라미터
알파 (폭 스케일링 계수), 베타 (깊이 스케일링 계수), 감마 (해상도 스케일링 계수)와 같은 모델 파라미터는 EfficientNet 모델을 구성하는 데 중요합니다. 이러한 파라미터는 전체 모델 크기와 계산 비용을 결정합니다. 이 구현 과정에서는 각 모델 변형에 대한 최적의 설정을 찾기 위해 이러한 파라미터에 대해 다양한 값들을 실험했습니다.

## 훈련과 평가
훈련을 진행할 시 parameters.ipynb 파일 내에 있는 train_model 함수를 사용해도 되고 직접 만들어서 사용이 가능합니다.
평가 시에는 parameters.ipynb 파일 내에 있는 test_model 함수를 사용해도 되고 직접 만들어서 사용이 가능합니다.

## 결과
이 모델을 CIFAR-10 데이터 셋과 CIFAR-100 데이터 셋을 이용해 torchvision 패키지에 있는 EfficientNet과 비교해보았습니다.


