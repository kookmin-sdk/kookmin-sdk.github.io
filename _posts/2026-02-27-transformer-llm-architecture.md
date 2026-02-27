---
layout: libdoc/page
title: "Transformer 및 대규모 언어 모델 아키텍처: 설계 원리와 구현"
date: 2026-02-27
category: Technical Deep-Dives
excerpt: "Transformer 아키텍처의 근간과 LLM의 발전, 그리고 DeepSeek 같은 최신 모델의 기술 혁신을 다룬 포괄적인 분석"
---

# Transformer 및 대규모 언어 모델 아키텍처: 설계 원리와 구현

## 서론

최근 인공지능 분야에서는 대규모 언어 모델(Large Language Model, LLM)을 비롯한 거대 매개변수 모델들의 발전이 혁신적인 성과를 보이고 있다.   
수백억에서 수천억 개의 매개변수를 가진 LLM들은 방대한 말뭉치로 사전 학습되어 텍스트 생성, 질의 응답, 코드 생성 등 다양한 작업에서 인간 수준에 가까운 성능을 보여주고 있다.  
  
2022년 공개된 ChatGPT는 사용자 질의에 유창하고 일관된 답변을 생성함으로써 전 세계적인 관심을 불러일으켰고, 이를 계기로 OpenAI, Google, Meta 등의 기업들이 경쟁적으로 초거대 언어 모델 개발에 투자하기 시작했다.  
생성 AI 붐 속에서, Meta가 공개한 LLaMA 모델과 DeepSeek 등이 대표적인 오픈소스 LLM으로 부상하였다.  
예를 들어, 중국의 AI 스타트업 DeepSeek는 2025년 자사 모델 DeepSeek-R1을 공개하여 미국 앱스토어 무료 앱 1위를 달성하고 ChatGPT의 인기를 뛰어넘는 등 큰 반향을 일으켰다.  
DeepSeek의 공개로 인해 엔비디아(NVIDIA)의 주가가 단기간 17% 하락했다는 보고도 있으며, 이는 개방형 LLM의 등장이 산업계에 미치는 파급력을 보여준다.

#### 이러한 대규모 AI 모델의 성공 이면에는 Transformer 아키텍처의 혁신적인 등장이 있었다.

2017년 Vaswani 등이 제안한 Transformer 모델은 이전까지 주로 사용되던 순환신경망(RNN)의 한계를 뛰어넘어, 셀프-어텐션(self-attention) 메커니즘을 통해 시퀀스 내 모든 단어 쌍 간의 상호작용을 한 번에 모델링할 수 있게 했다.  
RNN 계열이 갖고 있던 장기 의존성 학습의 어려움(예: 장기간 시퀀스에서 기울기 소실 문제)을 해결하기 위해, Transformer는 각 토큰이 전체 입력 문맥의 어떤 토큰에 주목해야 하는지를 학습하는 어텐션 기법을 도입하였다.  
그 결과, 멀리 떨어진 단어들 간의 관계까지 효율적으로 포착할 수 있게 되었고, 복잡한 패턴 학습 능력이 크게 향상되었다.

Transformer는 순환 구조가 없기 때문에 병렬 연산이 가능해 학습 속도 면에서도 RNN보다 유리하며, 이러한 이점들로 인해 현재 거의 모든 최신 LLM의 기본 토대가 되고 있다.  
사실상 "대규모 언어 모델 = Transformer의 확장판"이라고 해도 과언이 아닐 정도로, GPT 시리즈와 BERT 등 현대 NLP의 핵심 모델들은 모두 Transformer에 기반하고 있다.

#### 본 글에서는 Transformer와 LLM 아키텍처의 근간을 이루는 설계 철학과 기술적 발전을 학계 수준에서 상세히 고찰한다.

먼저 관련 연구 및 배경으로서 다양한 AI 시스템에서의 아키텍처 설계와 LLM으로의 패러다임 전환을 살펴본다.  
이후 Transformer 모델의 내부 구조를 집중적으로 설명하고, 핵심 구성요소인 다중-헤드 어텐션과 피드포워드 네트워크 등을 분석한다.  
또한 오픈소스 구현 예시로서 간단한 C++ 코드로 Transformer 블록의 동작을 구현해보며, Meta의 llama.cpp와 같은 최적화된 LLM 엔진이 어떻게 동작하는지도 언급한다.  
  
마지막으로 DeepSeek 모델을 비롯한 최신 LLM에서 도입된 새로운 기법들 – 예를 들어 Multi-head Latent Attention, Mixture-of-Experts, Multi-Token Prediction, Group Relative Policy Optimization 등 – 을 소개하고, 이러한 기술 혁신이 대규모 모델의 성능 및 효율 향상에 가져온 영향과 향후 전망을 논의한다.  
  
본 논문의 내용은 대규모 언어 모델의 기반이 되는 아키텍처 원리부터 실제 구현에 이르는 과정을 포괄적으로 다룬다.

---

## 관련 연구

아키텍처 설계는 인공지능 시스템 전반에 걸쳐 성능과 특성을 결정하는 핵심 요소입니다.  
예를 들어 자율주행 및 첨단 운전자 보조 시스템(ADAS) 분야에서는 차량의 여러 서브시스템을 통합적으로 제어하기 위한 계층적 제어 아키텍처가 연구되었습니다.  
Lin 등은 조향 및 제동과 같은 개별 기능들을 조정하여 차량의 거동을 관리하는 모션 관리 통합 아키텍처를 제안하였는데, 이를 통해 기존에 각각 작동하던 시스템 간 협조를 향상시켜 주행 경로 추종 오차를 줄일 수 있음을 보였습니다.  
이러한 협조 제어 아키텍처는 차량 시스템을 "시스템들의 시스템"으로 간주하여, 복잡한 기능들을 모듈화하고 상위 계층에서 조율함으로써 성능 개선을 달성합니다.

한편 휴먼-스웜 상호작용 분야에서는 투명성(Transparency)과 신뢰를 향상시키기 위한 아키텍처 개념이 제시되었습니다.  
Hepworth 등은 인간과 로봇 군집(swarm)이 협업할 때 설명 가능성(Explicability), 해석 가능성(Interpretability), 예측 가능성(Predictability)을 체계적으로 구현하기 위한 HST³ 아키텍처를 제안하였습니다.  
이 아키텍처는 투명성을 상위 개념으로 정의하고, 그 하위에 해석·설명·예측 가능성의 틀을 두어 인간-에이전트 팀의 상황 인식과 신뢰 형성을 도모합니다.  
이는 단순한 성능 향상을 넘어, AI 시스템의 이해력과 신뢰성을 구조적으로 높이려는 노력의 일환이라 할 수 있습니다.

이처럼 시스템 아키텍처는 자율주행 차량부터 인간-AI 협업에 이르기까지 다양한 맥락에서 중요합니다.  
딥러닝 모델 아키텍처 역시 지난 수십 년간 큰 변화를 겪어왔습니다.

과거에는 순환 신경망(RNN)과 LSTM 같은 구조가 시퀀스 모델링의 주류였으나, 심층 RNN은 긴 시퀀스에서 장기 의존성 학습에 어려움이 있었습니다.  
이를 극복한 Transformer의 등장은 딥러닝 패러다임의 전환점이 되었고, 이후 사전 학습 언어 모델의 시대를 열었습니다.  
2018년 등장한 BERT(Bidirectional Encoder Representations from Transformers)와 GPT(Generative Pre-trained Transformer)는 방대한 비지도 데이터로 사전 학습한 후 다운스트림 과제를 미세조정(fine-tuning)하는 접근으로 거의 모든 NLP 문제의 성능 지평을 끌어올렸습니다.  
특히 GPT 계열 모델은 디코더 기반 Transformer를 활용하여 대용량 텍스트를 생성하는데 특화되었고, BERT는 인코더 기반 Transformer로 문장의 이해 및 분석에 주로 사용되었습니다.  
이들 파운데이션 모델의 성공으로, 대규모 사전 학습→미세조정이라는 표준 기술 루틴이 정착되었습니다.

#### 대규모 언어 모델(LLM)은 이러한 사전 학습 Transformer 모델을 극한까지 확장한 형태입니다.

파라미터 수가 수십억→수백억→수천억으로 기하급수적으로 증가함에 따라 모델의 표현력도 크게 향상되었지만, 이에 비례하여 데이터 및 연산 자원의 필요량도 폭증하였습니다.  
최신 LLM의 학습은 수천 개의 GPU/TPU로 구성된 분산 시스템에서 수행되며, 하나의 모델을 학습하는데 수백만 달러에 달하는 비용이 소요되기도 합니다.  
또한 모델 용량의 증가와 함께, 단순한 확률적 언어 모델을 넘어 사용자 의도에 따라 응답을 조율하는 기법이 중요해졌습니다.  
OpenAI는 GPT-3 이후 인간 피드백을 통한 강화학습(RLHF)을 도입하여 모델의 응답이 보다 유용하고 안전하도록 훈련시켰습니다.  
RLHF는 인간 평가자가 선호하는 출력을 보상으로 주는 정책 최적화로, 기존의 언어 모델을 대화형 비서로 탈바꿈시킨 핵심 기술입니다.  
다만 표준 RLHF에서는 Proximal Policy Optimization(PPO) 알고리즘이 사용되고 보상함수에 KL 벌점을 추가하는 등 복잡한 설계가 들어갑니다.

#### 오픈소스 LLM의 부상도 주목할 만한 동향입니다.

2023년 Meta가 공개한 LLaMA는 연구 목적의 모델 가중치를 공개함으로써, 거대 언어 모델 연구의 민주화를 촉진했습니다.  
이를 기반으로 다양한 파생 모델들이 공개되고, Stanford Alpaca와 같이 비교적 저렴한 비용으로도 모델을 미세조정하는 사례가 나왔습니다.  
llama.cpp와 같은 프로젝트는 LLaMA 등의 모델을 일반 PC나 모바일 기기에서 실행 가능하도록 극도로 최적화한 C/C++ 구현체를 제공하였습니다.  
llama.cpp는 별도의 딥러닝 프레임워크 없이 순수 C/C++로 구현되었으며, Apple Silicon Neon 최적화, x86 AVX/AVX2 SIMD 최적화, NVIDIA GPU용 CUDA 커널 등을 통해 다양한 하드웨어에서 효율적으로 동작합니다.   

특히 4비트 이하 저정밀도 정수 양자화(quantization) 기법(1.5-bit, 2-bit, 3-bit, 4-bit 등)을 적용해 메모리 사용량을 획기적으로 감축하면서 추론 성능을 유지하는 데 성공하였습니다.  
이러한 최적화 덕분에 일반 소비자 GPU나 CPU만으로도 수십억~수백억 매개변수 모델을 실행할 수 있으며, 실제로 llama.cpp는 LLaMA(1,2), Mistral, Falcon, GPT-NeoX, 그리고 DeepSeek를 포함한 최신 공개 모델들까지 폭넓게 지원하고 있습니다.  

#### DeepSeek 모델의 가중치 역시 HuggingFace를 통해 공개되어, llama.cpp와 같은 엔진으로 누구나 실행해볼 수 있는 환경이 마련되었습니다.

---

## Transformer 모델 아키텍처

Transformer는 대규모 언어 모델의 근간을 이루는 딥러닝 모델 아키텍처로서, 인코더-디코더 구조와 셀프 어텐션 메커니즘을 특징으로 합니다.  
그림 1은 Transformer의 전형적인 아키텍처를 보여줍니다. 왼쪽은 인코더 블록을, 오른쪽은 디코더 블록을 나타내며, 각 블록은 여러 층(layer)의 반복으로 구성됩니다.  

> **그림 1: Transformer 디코더 블록의 구조 예시**  

<!-- ![Transformer Decoder Architecture](/assets/img/decoder.png) -->

각 층은 두 개의 서브층(sub-layer)으로 나뉘는데, 첫 번째 서브층은 멀티-헤드 어텐션이고 두 번째 서브층은 피드포워드 신경망(FFN)으로 이루어집니다.  
각 서브층 출력에는 잔차 연결(residual connection)이 적용되고, 그 뒤에 레이어 정규화(layer normalization)가 따라붙는 구조가 사용됩니다.  
이러한 디자인(잔차 연결 + 정규화)은 딥네트워크 학습 안정화와 그레이디언트 흐름 개선에 기여합니다.

Transformer의 인코더와 디코더 모두 이러한 기본 층 구조를 따르지만, 디코더의 경우 첫 번째 어텐션 서브층에서 마스킹(self-attention with masking) 기법을 사용하여 미래 토큰을 볼 수 없도록 함으로써 단계별 생성(auto-regressive generation)이 가능하도록 합니다.  
반면 인코더는 입력 전체 토큰을 한꺼번에 self-attention하기 때문에 마스킹이 필요 없습니다.

Transformer의 등장 당시에는 인코더-디코더 구조가 기계 번역 등 시퀀스 변환 문제에 사용되었으나, 이후 GPT 계열 모델들은 디코더 부분만을 활용하여 시퀀스 생성에 집중하였습니다.  
반대로 BERT 계열은 인코더 부분만으로 양방향(contextual) 언어 이해에 특화된 모델을 구성하였습니다.  
이처럼 응용 목적에 따라 Transformer 아키텍처의 일부를 취사선택하여 사용하지만, 기본 구성 요소와 동작 원리는 모두 동일합니다.

Transformer는 인코더-디코더 구조와 셀프 어텐션 메커니즘을 특징으로 합니다.

- **인코더/디코더 블록**: 여러 층(layer)으로 구성, 각 층은 멀티-헤드 어텐션과 피드포워드 신경망(FFN) 서브층으로 이루어짐  
- **잔차 연결 + 레이어 정규화**: 학습 안정화 및 그레이디언트 흐름 개선  
- **마스킹**: 디코더에서는 미래 토큰을 볼 수 없도록 마스킹 적용  
- **응용**: GPT 계열은 디코더만, BERT는 인코더만 사용

---

## 셀프-어텐션 메커니즘

#### Transformer의 핵심 혁신은 셀프-어텐션(self-attention) 메커니즘입니다.

셀프-어텐션은 한 층의 입력 시퀀스 내 각 토큰이 다른 모든 토큰과의 상관관계에 따라 자신을 표현하도록 변환하는 연산입니다. 이전 RNN에서는 각 시간 스텝에서 바로 이전 상태만 참고했다면, 어텐션을 통해 모든 위치의 입력을 가중합하여 출력 상태를 계산할 수 있게 되었습니다.

이를 구현하기 위해 Transformer는 각 토큰의 입력 벡터 $x$로부터 세 종류의 잠재 표현을 얻습니다: 쿼리(Query) $q$, 키(Key) $k$, 밸류(Value) $v$입니다.  
구체적으로는 학습 가능한 가중치 행렬 $W^Q, W^K, W^V$를 사용하여 $q = xW^Q$, $k = xW^K$, $v = xW^V$로 변환합니다.   

이렇게 하면 시퀀스의 모든 토큰에 대해 쿼리, 키, 밸류 벡터 집합 $Q, K, V$를 얻을 수 있습니다.

다음 단계로, 각 쿼리 $q_i$ (시퀀스의 $i$번째 토큰에 대응하는 쿼리)가 모든 다른 토큰 $j$의 키 $k_j$와 유사도 점수(score)를 구합니다. 이 유사도 점수는 보통 $q_i$와 $k_j$의 내적(dot product)으로 계산하며, $\text{score}_{ij} = q_i \cdot k_j / \sqrt{d_k}$의 형태로 정규화됩니다. 여기서 $d_k$는 키 벡터의 차원이고, 내적 값에 $\sqrt{d_k}$로 나누는 것은 스케일 조정을 통해 매우 큰 값이 나오지 않도록 하기 위함입니다(이를 스케일드 닷프로덕트 어텐션이라 합니다).

그 다음, 각 쿼리에 대한 점수 벡터 $\text{score}_i$에 소프트맥스(softmax) 함수를 적용하여 합이 1이 되는 확률 분포로 변환합니다. 즉, 토큰 $i$가 시퀀스 내 다른 토큰 $j$에 얼마만큼 주의를 기울여야 하는지를 확률적으로 구하는 셈입니다. 이 확률을 어텐션 가중치 $\alpha_{ij}$라고 하면, 해당 출력은 모든 밸류 벡터의 가중합으로 계산됩니다.  

$$\text{out}_i = \sum_j \alpha_{ij} v_j$$

이렇게 함으로써, 토큰 $i$의 출력 표현은 자신과 관련성이 높은 다른 토큰들의 정보가 강조된 형태로 변환됩니다. 
예를 들어, 번역 모델에서 영어 문장의 단어 $i$가 프랑스어 문장 단어 $j$와 강하게 매칭된다면, $i$의 쿼리는 $j$의 키와 큰 내적값을 가져서 높은 어텐션 가중치를 할당받고, $v_j$ (단어 $j$의 의미 벡터)가 $i$의 출력에 크게 기여하게 됩니다.

#### Transformer에서는 이러한 어텐션 연산을 하나의 헤드(head)만 사용하는 것이 아니라 여러 헤드에 걸쳐 병렬로 수행합니다.

이를 멀티-헤드 어텐션(Multi-Head Attention)이라고 부릅니다. 예컨대 8개 헤드를 사용한다면, 각 헤드는 입력을 서로 다른 $W^Q_h, W^K_h, W^V_h$ 가중치를 통해 각기 다른 임베딩 공간에서 어텐션을 수행합니다. 어떤 헤드는 문법적인 연관성에 집중할 수 있고, 다른 헤드는 의미적 연관성에 집중하는 식으로, 헤드마다 다양한 종류의 관계를 학습하게 됩니다.

각 헤드의 출력은 길이 $d_v$인 벡터인데(보통 $d_v = d_k$로 설정), 이를 모두 연결(concatenate)하여 $d_{\text{model}}$ 길이의 벡터로 만들고, 마지막으로 별도 가중치 $W^O$를 곱해 다시 모델 차원으로 투영(projection)합니다. 이 $W^O$까지 적용한 최종 결과가 멀티-헤드 어텐션 층의 출력이 됩니다. 수식으로 요약하면, 어텐션 층 출력은  

$$\text{Attention}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d_k})\,V$$

로 표현할 수 있으며, Multi-Head의 경우 여러 어텐션 결과를 합친 뒤 $W^O$를 적용합니다.

마지막으로, Transformer 모델은 위치 정보를 RNN처럼 순차적으로 처리하지 않으므로, 토큰의 위치 인코딩(positional encoding)을 추가로 사용합니다. 원 논문에서는 사인(sin)과 코사인(cos) 함수를 주기적으로 변조한 고정된 위치임베딩을 사용하였고, 이후 Learnable positional embedding 등 다양한 변형이 사용되고 있습니다. 위치 인코딩은 각 토큰 벡터에 더해져서, 모델이 각 단어의 순서나 상대적 거리를 인식할 수 있도록 합니다.

셀프-어텐션은 각 토큰이 다른 모든 토큰과의 상관관계에 따라 자신을 표현하도록 변환합니다.

1. **Query, Key, Value 계산**: 입력 벡터 $x$로부터 $q = xW^Q$, $k = xW^K$, $v = xW^V$로 변환  
2. **유사도 점수 계산**: $q_i$와 $k_j$의 내적을 통해 score 산출, $\sqrt{d_k}$로 정규화  
3. **Softmax 적용**: 어텐션 가중치 $\alpha_{ij}$ 계산  
4. **출력 계산**: $\sum_j \alpha_{ij} v_j$로 각 토큰의 출력 표현 생성  
5. **멀티-헤드 어텐션**: 여러 헤드에서 병렬로 어텐션 수행, 다양한 관계 학습  
6. **위치 인코딩**: 토큰 순서 정보를 위해 사인/코사인 기반 또는 학습형 임베딩 추가

---

## 피드포워드 네트워크 및 기타 구성요소

멀티-헤드 어텐션 층의 출력은 각 위치(token)에 대한 문맥적 표현입니다. Transformer는 여기에 위치별 완전연결 신경망을 적용하여 각 위치의 표현을 더욱 변환합니다. 이를 피드포워드 신경망(FFN) 서브층이라고 하며, 일반적으로 두 개의 밀집층으로 구성됩니다. 첫 번째 밀집층은 차원을 확장하며(보통 $4 \times d_{\text{model}}$ 차원으로 확대), 비선형 활성함수(ReLU 등)를 적용한 뒤, 두 번째 밀집층에서 다시 원래 차원 $d_{\text{model}}$로 축소합니다.

공식으로 나타내면 각 위치 $i$에 대해:  
$$\text{FFN}(x_i) = \phi(x_i W_1 + b_1)\, W_2 + b_2$$

여기서 $\phi$는 ReLU와 같은 활성함수, $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$이며 $d_{\text{ff}}$는 확장된 내부 차원입니다. 원 논문에서는 $d_{\text{ff}} = 4\,d_{\text{model}}$로 설정하여 차원을 4배로 늘렸다가 다시 줄였으며, 활성함수로 ReLU를 사용하였습니다.

이 FFN 서브층은 토큰별로 독립적으로 적용되지만(시퀀스 상의 각 위치에 동일한 두 개의 밀집층이 적용), 어텐션 층을 통해 각 위치에 다른 위치의 정보가 섞여들어간 상태이므로 간접적으로 토큰 간 상호작용이 이어지는 효과가 있습니다.

Transformer의 한 층은 앞서 설명한 멀티-헤드 어텐션 서브층 + FFN 서브층으로 이루어집니다. 각 서브층 출력에는 입력을 더해주는 잔차 연결이 적용되고, 그 합에 대해 Layer Normalization을 수행합니다(원래 논문은 후-정규화(post-LN) 사용, 이후 학습 안정성을 위해 사전-정규화(pre-LN) 구조도 널리 쓰임). 이러한 과정을 거쳐 출력된 각 층의 결과를 다음 층의 입력으로 보내고, 여러 층을 스택(stack)하여 모델의 깊이를 형성합니다.

예컨대 12층 Transformer 디코더를 사용한 GPT-2, 24층의 GPT-3, 32층의 PaLM, 40~80층의 GPT-4 등으로 계속 깊어지는 추세입니다. 층의 개수와 헤드 수, 차원 등은 모델 크기에 따라 달라지며, 이들을 모델 하이퍼파라미터라고 합니다.

요약하면, Transformer 아키텍처는 (어텐션 + FFN)의 층을 여러 번 반복하여 심층 표현을 학습하는 구조로, 입력 시퀀스를 병렬로 처리하면서도 토큰 간 복잡한 상호의존 관계를 효과적으로 포착한다는 점에서 딥러닝의 강력한 기반이 되었습니다.

- **FFN 서브층**: 각 위치별로 두 개의 밀집층 적용, 차원 확장 후 축소, 활성함수(ReLU 등) 사용  
- **잔차 연결 및 레이어 정규화**: 각 서브층 출력에 적용  
- **층 스택**: 여러 층을 반복하여 심층 표현 학습

---

## 구현 예시 및 동작 분석

Transformer의 self-attention 연산이 실제로 어떻게 동작하는지 이해를 돕기 위해, 간단한 C++ 구현을 통해 예시를 보여준다. 
아래 코드는 소규모 입력에 대해 멀티-헤드 어텐션 한 층을 수행하는 프로그램이다.   

#### 모델 차원 $d_{\text{model}}=6$, 헤드 수 $N_{\text{head}}=2$

로 가정하여 각 헤드의 차원 $d_k = 3$이며, 토큰 시퀀스 길이는 3으로 설정하였다. 가중치 행렬 $W^Q, W^K, W^V, W^O$는 임의의 값으로 초기화하였다. 

코드에서는 먼저 입력 행렬 X에 각 헤드별로 $W^Q, W^K, W^V$를 곱하여 $Q, K, V$ 행렬을 계산한 뒤, 어텐션 스코어 행렬과 softmax를 통해 헤드별 출력을 구하고, 마지막으로 두 헤드 출력을 concat하여 $W^O$를 적용함으로써 최종 출력을 산출한다.

```cpp
#include <iostream>
#include <cmath>
using namespace std;

const int SEQ_LEN = 3; // 시퀀스 길이 (토큰 개수)
const int D_MODEL = 6; // 모델 임베딩 차원 (d_model)
const int N_HEAD = 2;  // 헤드 개수 (multi-head attention)
const int D_K = D_MODEL / N_HEAD; // 한 헤드당 임베딩 차원 (d_k)

// 예시 입력 (토큰 임베딩 행렬 X[SEQ_LEN][D_MODEL])
float X[SEQ_LEN][D_MODEL] = {
    {1.0, 0.5, -1.0, 2.0, 0.0, -0.5},
    {0.3, -0.7, 0.8, -0.1, 1.0, 0.2},
    {0.0, 0.4, 0.5, -0.5, -0.2, 0.3}
};

// 가중치 행렬들 (Wq, Wk, Wv: 각 헤드의 Query/Key/Value 변환, Wout: 출력 투영)
float Wq[N_HEAD][D_MODEL][D_K];
float Wk[N_HEAD][D_MODEL][D_K];
float Wv[N_HEAD][D_MODEL][D_K];
float Wout[D_MODEL][D_MODEL];

// 가중치 초기화 (예제에서는 단순한 패턴 값으로 초기화)
void init_weights() {
    for(int h = 0; h < N_HEAD; ++h) {
        for(int i = 0; i < D_MODEL; ++i) {
            for(int j = 0; j < D_K; ++j) {
                // 단순 패턴: Wq[h][i][j] = 0.1 * (h+1) * (i+1)
                Wq[h][i][j] = 0.1f * (h+1) * (i+1);
                // Wk, Wv도 유사한 방식으로 패턴 값 부여
                Wk[h][i][j] = 0.05f * (h+1) * (i+1);
                Wv[h][i][j] = -0.03f * (h+1) * (i+1);
            }
        }
    }
    // 출력 프로젝션 Wout을 단위행렬(identity)로 초기화 (편의상)
    for(int i = 0; i < D_MODEL; ++i) {
        for(int j = 0; j < D_MODEL; ++j) {
            Wout[i][j] = (i == j ? 1.0f : 0.0f);
        }
    }
}

int main() {
    init_weights();

    // Q, K, V 행렬을 저장할 배열 (각 헤드마다 SEQ_LEN x D_K 크기)
    float Q[N_HEAD][SEQ_LEN][D_K];
    float K[N_HEAD][SEQ_LEN][D_K];
    float V[N_HEAD][SEQ_LEN][D_K];

    // 1. 입력 X를 각 헤드의 Query, Key, Value 공간으로 선형 변환
    for(int h = 0; h < N_HEAD; ++h) {
        for(int t = 0; t < SEQ_LEN; ++t) {
            for(int j = 0; j < D_K; ++j) {
                float q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;
                for(int i = 0; i < D_MODEL; ++i) {
                    q_sum += X[t][i] * Wq[h][i][j];
                    k_sum += X[t][i] * Wk[h][i][j];
                    v_sum += X[t][i] * Wv[h][i][j];
                }
                Q[h][t][j] = q_sum;
                K[h][t][j] = k_sum;
                V[h][t][j] = v_sum;
            }
        }
    }

    // 2. 각 헤드에 대해 어텐션 스코어와 헤드별 출력 계산
    float head_out[N_HEAD][SEQ_LEN][D_K];
    for(int h = 0; h < N_HEAD; ++h) {
        // 어텐션 스코어 행렬 계산 (score[t][s] = Q_h[t] · K_h[s]^T / sqrt(d_k))
        float score[SEQ_LEN][SEQ_LEN];
        for(int t = 0; t < SEQ_LEN; ++t) {
            for(int s = 0; s < SEQ_LEN; ++s) {
                // Q_h의 t번째 토큰 벡터와 K_h의 s번째 토큰 벡터 내적
                float dot = 0.0f;
                for(int r = 0; r < D_K; ++r) {
                    dot += Q[h][t][r] * K[h][s][r];
                }
                score[t][s] = dot / sqrt((float)D_K); // 스케일링
            }
        }
        // softmax 적용하여 어텐션 가중치 계산
        float softmax_score[SEQ_LEN][SEQ_LEN];
        for(int t = 0; t < SEQ_LEN; ++t) {
            // 각 행에 대해 softmax
            float max_val = score[t][0];
            for(int s = 1; s < SEQ_LEN; ++s) {
                if(score[t][s] > max_val) max_val = score[t][s];
            }
            float sum_exp = 0.0f;
            for(int s = 0; s < SEQ_LEN; ++s) {
                softmax_score[t][s] = exp(score[t][s] - max_val);
                sum_exp += softmax_score[t][s];
            }
            for(int s = 0; s < SEQ_LEN; ++s) {
                softmax_score[t][s] /= sum_exp;
            }
        }
        // softmax 가중치를 이용하여 Value들의 가중합 계산 (헤드별 출력)
        for(int t = 0; t < SEQ_LEN; ++t) {
            for(int r = 0; r < D_K; ++r) {
                float sum = 0.0f;
                for(int s = 0; s < SEQ_LEN; ++s) {
                    sum += softmax_score[t][s] * V[h][s][r];
                }
                head_out[h][t][r] = sum;
            }
        }
    }

    // 3. 여러 헤드의 출력을 이어붙이고 최종 선형 변환으로 출력 생성
    float output[SEQ_LEN][D_MODEL];
    for(int t = 0; t < SEQ_LEN; ++t) {
        // 각 헤드 출력 벡터를 concat하여 concat_vec 생성 (길이 D_MODEL)
        float concat_vec[D_MODEL];
        for(int h = 0; h < N_HEAD; ++h) {
            for(int r = 0; r < D_K; ++r) {
                concat_vec[h * D_K + r] = head_out[h][t][r];
            }
        }
        // concat_vec에 출력 가중치 Wout 적용 (선형 결합)
        for(int j = 0; j < D_MODEL; ++j) {
            float sum = 0.0f;
            for(int i = 0; i < D_MODEL; ++i) {
                sum += concat_vec[i] * Wout[i][j];
            }
            output[t][j] = sum;
        }
    }

    // (옵션) 결과 출력 - 각 토큰에 대한 최종 임베딩 벡터
    for(int t = 0; t < SEQ_LEN; ++t) {
        cout << "Token " << t << " output: ";
        for(int j = 0; j < D_MODEL; ++j) {
            cout << output[t][j] << " ";
        }
        cout << endl;
    }
    return 0;
}
```

**설명**: 입력 행렬 X에 대해 각 헤드별로 Query/Key/Value 계산, 어텐션 스코어 및 softmax, 헤드별 출력 생성, 최종 출력 벡터 산출 과정을 구현합니다. 실제 대규모 모델은 벡터화 연산과 최적화를 통해 GPU에서 병렬 수행합니다. llama.cpp 등 오픈소스 엔진은 CPU/GPU 최적화 및 양자화로 대규모 모델을 로컬에서 실행할 수 있게 합니다.

**코드 분석 및 실전 활용**: 위 C++ 예제는 2-head, $d_{\text{model}}=6$ 환경에서 Multi-Head Self-Attention 연산의 핵심 과정을 보여줍니다. 각 단계에서 Query/Key/Value 계산, 어텐션 스코어 및 softmax 적용, 그리고 최종 출력 벡터 생성까지의 흐름을 명확히 구현하였습니다. 실행 시 `Token 0 output: ...` 형태로 각 토큰별 결과 벡터가 출력됩니다.  

실제 대규모 모델에서는 배열과 이중 루프 대신, PyTorch나 TensorFlow와 같은 프레임워크의 고성능 행렬 연산 라이브러리를 활용하여 GPU에서 대량 병렬 처리를 수행합니다. 오픈소스 프로젝트인 llama.cpp는 어텐션 및 FFN 연산을 CPU 벡터화와 양자화로 최적화하여, $L$개의 Transformer 층과 수십억 매개변수를 가진 LLM도 노트북이나 모바일 기기에서 실행할 수 있도록 지원합니다.  

llama.cpp는 Apple Neural Engine, x86 SIMD, NVIDIA CUDA 등 다양한 하드웨어에 맞춘 저수준 최적화를 적용하여 최신 LLM의 추론을 경량화하였으며, 이를 통해 연구자와 개발자는 클라우드 서버 없이도 대규모 모델을 로컬 환경에서 실험할 수 있습니다. 이 엔진 덕분에 LLaMA, GPT-2/3, Mistral, StableLM, DeepSeek 등 다양한 LLM을 단일 인터페이스에서 실행할 수 있어, 접근성과 확장성 연구에 크게 기여하고 있습니다.

---

## DeepSeek 모델의 기술적 혁신

DeepSeek는 최신 LLM에서 다음과 같은 주요 기술을 도입했습니다.

- **MLA (Multi-head Latent Attention)**: Key/Value 행렬을 저차원 잠재 공간에 압축/복원하여 메모리 사용 절감
- **MoE (Mixture-of-Experts)**: FFN을 여러 전문가 네트워크로 분할, 일부 전문가만 선택적으로 활성화하여 효율 및 표현력 향상
- **MTP (Multi-Token Prediction)**: 한 번에 여러 토큰을 병렬로 예측, 생성 속도 향상
- **GRPO (Group Relative Policy Optimization)**: PPO 기반 RLHF의 대안, 가치망 제거 및 KL 정규화 손실 직접 포함, 그룹 어드밴티지 도입

추가적으로 FP8 혼합 정밀도 훈련, 분산 학습 프레임워크 등 공학적 최적화도 적용되었습니다.

---

## 결론 및 향후 전망

Transformer 기반 LLM의 발전은 AI 성능 도약의 원동력이었습니다. 앞으로는 다음과 같은 방향이 중요해질 것입니다.

- **효율성**: 모델 압축, 어텐션 최적화, 필요한 계산만 수행하는 아키텍처 연구
- **다중모달 통합**: 텍스트 외 이미지, 음성, 영상 등 복합 입력을 처리하는 멀티모달 모델
- **신뢰성과 투명성**: 설명 가능 인공지능(XAI), 모델 내부 분석, 인간 중심 디자인
- **데이터 및 훈련 패러다임 혁신**: 자기지도학습, 순수 RL, 명시적 추론 능력 부여 등

Transformer와 LLM 아키텍처는 계속 진화하고 있으며, 더 똑똑하고 효율적이며 신뢰할 수 있는 AI를 만들기 위한 연구가 활발히 진행되고 있습니다.

---

## 참고문헌

1. **Lin, T., Ji, S., Dickerson, C. E., & Battersby, D.**  
    "Coordinated Control Architecture for Motion Management in ADAS Systems," *IEEE/CAA J. Autom. Sinica*, vol. 5, no. 2, pp. 432–444, Feb. 2018.

2. **Hepworth, A. J., et al.**  
    "Human-Swarm-Teaming Transparency and Trust Architecture," *IEEE/CAA J. Autom. Sinica*, vol. 8, no. 7, pp. 1281–1295, Jul. 2021.

3. **Xiong, L., et al.**  
    "DeepSeek: Paradigm shifts and technical evolution in large AI models," *IEEE/CAA J. Autom. Sinica*, vol. 12, no. 5, pp. 841–858, May 2025.

4. **Deng, Z., et al.**  
    "Exploring DeepSeek: A survey on advances, applications, challenges and future directions," *IEEE/CAA J. Autom. Sinica*, vol. 12, no. 5, pp. 872–893, May 2025.

5. **ggml-org, llama.cpp**  
    "LLM inference in C/C++," GitHub Repository, 2023.  
    [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)

6. **Wikipedia**  
    "Transformer (deep learning architecture)," Wikipedia, The Free Encyclopedia, 2023.

---

*이 논문은 Transformer 아키텍처의 기초부터 실제 구현과 최신 혁신까지 포괄적으로 다룹니다.*
