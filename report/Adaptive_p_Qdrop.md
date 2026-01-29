# Adaptive p for QDrop

## Motivation
Qdrop 기법의 목표는 가중치 주변의 손실 지형을 평평하게(Flatness) 만드는 것이다.

가중치를 튜닝하는 동안 각 Activation 값에 대해 확률 p로 양자화를 적용하고, 
1-p 확률로는 양자화를 하지 않고 그냥 FP32로 가정하고 가중치를 튜닝한다.

즉, 모든 원소에 대해 고정된 확률(p=0.5)를 적용한다는 것이다.
이는 정보 이론적으로 엔트로피가 최대화되는 지점, 즉 "가장 예측 불가능한 노이즈"를 만들기 위함이다.

CNN과 Transformer의 가중치와 활성화 값은 위치(Channel/Layer)마다 그 **중요도(Sensitivity)**와
**곡률(Hessian)**이 천차만별이다.
 - 어떤 채널은 조금만 노이즈가 섞여도 전체 정확도가 급락한다.(High Sensitivity)
 - 어떤 채널은 노이즈가 많이 섞여도 결과에 영향이 거의 없다.(Low Sensitivity)

Qdrop의 p=0.5 설정은 이러한 "채널 간의 불균형"을 무시하고 기계적으로 동일한 노이즈를 주입한다고 생각한다.
이는 민감한 채널에는 과도한 노이즈를 주어 학습을 방해하고, 둔감한 채널에는 불충분한 노이즈를 주어 평탄화 
기회를 놓치는 비효율성을 초래할 수 있다고 생각한다.

## Theorical Analysis
### 의문점 1. 가중치와 베르누이 분포의 관계

"가중치는 베르누이 분포와 어떤 관계이며, 논문의 가정이 타당한가?"

논문에서는 활성화 노이즈 u가 베르누이 분포 B(1, 0.5)를 따른다고 가정한다. 그리고 u가 선형 변환을 통해
가중치 섭동 v로 이식된다고 주장한다.

 - 논문의 논리: u가 랜덤하게 켜지고 꺼지므로(Bernoulli), 이에 대응하는 v도 랜덤한 방향성을 가진다.
   따라서, p=0.5일 때 노이즈 패턴의 다양성(Diversity)이 최대화되어 가장 많은 방향의 Flatness를 확보할 수 있다.

그렇다면, 다양성(Diversity)이 "최적화 성능(Optimality)"을 의미하는가? 즉, 모든 방향이 평등하다는 전제 하에 위 논리를 적용한 것이 아닌가? 실제로는 가중치 공간의 곡률이 찌그러져 있기 때문에, 뾰족한 방향으로는 노이즈를 적게(분산을 작게)주고, 평평한 방향으로는 노이즈를 많이 줘야 한다. 베르누이 분포 자체보다는 "그 분포의 파라미터 p를 어떻게 설정하느냐"가 핵심일 것이다.
 - 최적화 관점(Taylor Expansion)에서 손실 함수의 기댓값은 $E[\Delta L] \approx \sum H_{ii}\sigma_i^2$ 꼴로 나타난다.(H는 Hessian, $\sigma$는 노이즈 분산)
 - 여기서 노이즈 분산은 1-p에 비례한다.
 - 즉, H가 큰(민감한) 곳에서는 분산을 줄여야(즉, p를 높여야한다.) 손실을 최소화할 수 있다.
 - 따라서 무조건 적인 p=0.5는 수학적으로 최적해가 아니며, Hessian을 고려한 p의 조절이 필요하다.

### 의문점 2. 랜덤 p vs 민감도 기반 p
 "p를 그냥 랜덤하게 주건, 민감도를 계산해 주면 성능이 오를까?"

 - **Random p** : 단순히 p를 0.2 ~ 0.8 사이에서 랜덤하게 부여하는 것은 p=0.5보다 큰 이득이 없을 것 같다. 왜냐하면 중요하지 않은 채널을 보호하고 중요한 채널을 공격하는 '최악의 경우'가 발생할 수 있기 때문이다.
 - **Adaptive p based on Sensitivity** : 민감도에 따라 p를 조절하면 성능이 향상될 것이다.
    - **High Sensitivity Channel** : p(확률을 크게 준다.), Drop을 자주함(=FP32 유지) -> 가중치 보호, 학습 안정성 확보
    - **Low Sensitivity Channel** : p(확률을 작게 준다.), Quantize 자주하여 적극적인 노이즈 주입으로 Flatness를 극대화한다.

## Methodology
Sensitivity-Aware QDrop 

 - Sensitivity Metric 선정 계산 비용이 높은 Hessian 대신, 활성화 값의 크기(Magnitude)를 대리 지표(Proxy)로 사용한다.(일반적으로 값이 클수록 양자화 오차에 민감하다.- AWQ, SmoothQuant, Outlier Suppression)
 $$
 S_c = \frac{1}{N} \sum |A_{c,i,j}|
 $$

### Adaptive Probability Mapping 각 채널의 c의 민감도 S를 확률 p로 변환한다.
$$
p_c = p_{min} + (p_{max}-p_{min}) \times \text{Normalize}(S_c)
$$
 - 민감도가 높을수록 $p_c$를 높게 설정하여(최대 0.8) 해당 채널이 양자화 노이즈에 의해 망가지는 것을 방지한다.
 - 민감도가 낮을수록 $p_c$를 낮게 설정하여(최소 0.2), 더 많은 노이즈를 겪게 해 robustness를 높인다.

### Implementation Strategy
 - **Pre-computation** : 튜닝 시작 전, Calibration 데이터를 소량 흘려보내 $p_c$ 벡터를 미리 계산하고 고정한다.
 - **Efficient Masking** : 훈련 중에는 채널 별로 서로 다른 확률 $p_c$를 적용한 Bernoulli Mask를 생성하여 Element-wise 연산을 수행한다.

### 예상되는 결과
 - 기존 QDrop의 Uniform Randomness(p=0.5) 가정이 최적이 아님을 수학적으로 보이고, 채널별 중요도를 고려해야 함을 입증한다.
 - 성능 향상: 민감한 채널을 보호함으로써 초기 학습 안정성을 높이고, 결과적으로 기존 Qdrop 대비 2비트 등 초저정밀도 환경에서 더 높은 정확도를 달성한다.
 - 효율성: Hessian 계산 없이 Magnitude 기반의 Proxy를 사용하여, 기존 Qdrop과 거의 동일한 학습 시간으로 성능 개선을 이뤄낸다.

---
---
# Implementation

