---
layout: post
categories: 논문리뷰
tags: audio, ASR
comments: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] Squeezeformer: An efficient transformer for automatic speech recognition."
---

출처
> Kim, Sehoon, et al. "Squeezeformer: An efficient transformer for automatic speech recognition." *arXiv preprint arXiv:2206.00888* (2022).


Copyright of figures and other materials in the paper belongs to original authors.

# 0 Abstract

- The recently proposed Conformer model has become the de facto backbone model for various downstream speech tasks based on its hybrid attention-convolution architecture that captures both local and global features.
    - Conformer model은 speech task들의 여러가지 downstream task 에 backbone 모델으로 사용되어 왔음.
    - **하지만 본 연구에서는 Conformer의 구조는 optimal하지 않다는 점을 기반으로, Squeezeformer을 제안함.**
- Squeezeformer 은 **ASR model**에서 SOTA 급의 성능을 보임
- Macro-architecture
    - Squeezeformer은 **Temporal u-Net 구조**를 사용해서 **multi head attention modeule**을 긴 시퀀스에 적용할 때 발생하는 **계산 비용을 줄이**려고 함
    - multi head attention이나 convolution module 의 심플한 block 구조 뒤에 feed foward module을 적용시킨 구조를 사용함.
        - Conformer은 마카롱 구조를 사용
- Micro-architecture
    - Squeezeformer은 **convolutional block에 있는 activation을 simplify**해줌
    - **Layer normalizataion operation**의 redundant 를 감소시켜줌
    - 효과적인 **depthwise down sampling layer**을 사용해서 input signal의 sub sampling을 얻어냄
        - depthwise→pointwise conv 순서대로 적용

# 1 Introduction

- 일반적으로 **CNN model은 global context을 capture하는 데에 한계**가 있음
- **Transformer은 계산량이 많고 memory overhead가 발생한다는 한계점**이 있음
- Conformer은 CNN과 Transformer의 장점만 사용할 수 있도록한 구조로, ASR task뿐만 아니라 speech processing task에서 사용되는 가장 기본적인 모델이 됨
- **Conformer의 한계**는 다음과 같음
    1. attention mechanism으로 인해서 **long sequence length를 효과적으로 처리하지 못함**
    2. Conformer 구조는 상대적으로 transformer 보다 **복잡한 구조**를 들고 잇음
- **Squeezeformer의 의의**
    1. (특히 깊은 네트워크에서의) 이웃하는 speech 프레임들의 학습된 feature representation에서 높은 **temporal 반복성**을 발견했고, 이로 인해서 **불필요한 계산이 진행되는 것을 temporal U-Net 구조로 해결**함
    2. back to back Multi head attention과 conv module으로 구성된 **마카롱 구조가 suboptimal**하다는 점을 기반으로 좀 더 **간단한 block 구조**를 제안함
        1. 해당 논문에서 제안된 block sturcture 은 standard 한 transformer block과 유사
            1. Multi head attention과 conv module 뒤에 각각 feed forward module 이 붙는 구조

# 2 Related Work

## 2.1 Model architecture for end to end ASR

- 최근 ASR 모델들은 전형적으로 encoder과 decoder으로 구성되어 있음
    - encoder에서는 speech signal을 input으로 받고 high level acoustic feature을 추출함
    - decoder에서는 extract된 feature들을 text sequence으로 변환함
- 가장 유명한 backbone model의 구조는 CNN임
    - 하지만 CNN은 global 한 context를 capture하는 것에 실패함
- transformer 모델은 speech frame간의 long range dependencies를 포착할 수 있지만 계산량이 너무 많음
- Conformer의 hybrid attention convolution architecture 은 다양한 speech task에서 SOTA급 성능을 보였지만, attention layer의 엄청난 complexity는 한계로 남아있음
    - Multihead attention의 cost를 줄이기 위한 다양한 approach들이 있었지만 해당 연구들은 overall 구조를 바꾸는 것은 아니었음

## 2.2 Training Methodology for end to end ASR

- 지난 몇년 간 contrastive learning에 기반한 다양한 self supervised learning 방법과 masked prediction이 ASR 의 성능을 향상시키기 위해서 사용되어 왔음

# 3 Architecture Design

- Conformer vs Squeezeformer
    
    ![Figure 2: (Left) The Conformer architecutre and (Right) the Squeezeformer architecture which comprises of the Temporal U-Net structure for downsampling and upsampling of the sampling rate, the standard Transformer-style block structure that only uses Post-Layer Normalization, and the depthwise separable subsampling layer.](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/860b7afd-5a85-4b08-a8ae-147fd82d1a90){: width="80%" height="80%"}{: .center}
    
    Figure 2: (Left) The Conformer architecutre and (Right) the Squeezeformer architecture which comprises of the Temporal U-Net structure for downsampling and upsampling of the sampling rate, the standard Transformer-style block structure that only uses Post-Layer Normalization, and the depthwise separable subsampling layer.
    

## 3.1 Macro-Architecture Design

### 3.1.1 Temporal U-Net Architecture

- **attention mechanism 과 convolution 의 hybrid structure은 conformer 으로부터 global 과 local한 interaction들을 모두 capture할 수 있도록 도왔음**
- 하지만 **attention operation은 quadratic FLOPs complexity**를 가지고 있음
    - quadratic FLOPs complexity는 간단히 이해하자면 엄청나게 많은 연산량임
    - Conformer model에서는 input sampling rate을 conv subsampling block에서 10ms~40ms rate으로 줄여주었지만 해당 rate은 network 전체에서 동일한 값을 가지게 됨
    - 즉 모든 attention 과 conv 연산에 constant한 temporal scale이 적용된다는 것
- 본 연구에서는 각 speech frame의 learned feature embedding들이 conformer model depth 를 통하여 어떻게 미분되어지는지에 대해서 분석해볼 예정
    - 본 연구에서는 Librispeech의 audio signal 샘플 100개를 랜덤하게 추출해서 conformer block을 통해서 process하였고 conformer block activation 마다 값을 기록했음
    - 이웃하는 embedding vecotr간의 cosine 유사도를 측정한 결과 아래 그림과 같은 plot이 나옴
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/c38caee5-24a5-4702-953f-dd8c98c5d63a){: width="80%" height="80%"}{: .center}
        
    - 연구진들은 **각각의 speeech frame의 embedding 값과 바로 이웃하는 speech frame의 embedding 값이 거의 95%의 similarity 를 확인하였음.** 4 frame 정도 떨어진 speech frame들도 거의 simailarity가 80%가 넘어감을 확인함.
    - 이러한 결과는 network의 **conformer block이 깊어질수록 temporal redundancy가 커진다**고 이해할 수 있음.
    - 따라서 본 연구에서는 이러한 feature embedding vector에서 발생하는 redundancy는 불필요한 계산량을 요하게 되며 network이 깊어질수록 sequence의 길이가 reduced 될 수 있음을 가정함(accuracy의 loss없이)
- **본 연구의 첫번째 macro-architecture imporvement step에서는 embedding vector이 model의 early block으로부터 process되었을 때 subsempling할 수 있도록 conformer model을 바꿈**
    - particular하게 연구진들은 conformer의 7번째 block까지는 샘플링 속도를 40ms 으로 유지하였지만, 이후 레이어부터는 풀링 레이어를 사용하여 80ms 으로 속도를 설정하여 subsampling을 진행함.
    - pooling layer에서는 depthwise separable conv 를 stride 2, kernel 3 으로 설정하여 이웃한 embedding간의 redundancy를 합쳐줌
        - 아마도 이거 pooling을 통해서 temporal한 영역에서 발생하는 중복을 어느정도 해결해줬다는 의미인것 같음
        - 이를 통해서 attention complexity는 4X으로 줄어들 수 있었고 feature간의 redundancy 도 줄일 수 있었다고 함
- 하지만 temporal  downsampling을 그냥 적용했을 때는 학습이 unstable하고 diverge될 수 있다는 문제점이 발생하게 됨
    - 이러한 현상이 발생하는 이유는 80ms 으로 subsampling을 한 이후에 decoder으로부터 중분한 resolution을 받지 못했을 경우임.
    - decoder은 각 음성 프레임에 대한 임베딩을 단일 레이블에 mapping하기에 전체 sequence을 성공적으로 디코딩하기 위해서는 충분한 resolution이 필요함
        - CV 영역에서 U-Net이 성공적으로 dense prediction을 수행하는 것에 기반하여 본 연구에서는 **Temporal U-Net 구조를 통해서 resolution  문제를 해결하고자 함**
- **Temporal U-Net을 사용했을 때의 이점**
    - **Conformer에 비하여 FLOPs가 20% 줄어듦**
    - WER가 7.9%에서 7.28%으로 줄어듦
        - WER : Word Error Rate
    - **이웃하는 embedding값이 유사해지는 것을 어느정도 방지할 수 있음**

### 3.1.2 Transformer-Style Block

- Conformer Block은 FMCF(Feed Forward 의 sequence, Multi head attention, Convolution, FeedForward module)으로 구성됨.
    
    **Feed Forward module** + Multi head attention + Convolution + **Feed Forward module** 
    
    → 원래는 위와 같이 FFM이 MC를 안고 있는 마카롱 구조임 
    
    - 참고로 ASR model에 있는 conv 의 커널 사이즈는 흔히 CV task에서 사용되는 network의 conv 커널 사이즈보다 큼
- 이 때문에 conv와 multi head attention module을 같이 넣는게 좋은 선택은 아닐 것이라고 연구진들은 주장하며 MF/CF 구조를 제안함
    
    **Multihead attention** + Feed forward module  / **Convolution** + Feed Forward module
    
    → Multihead attention과 Conv가 비슷한 역할을 수행하니까, 연속적으로 이 둘을 붙여놓지 않고 떼어 놓음
    
- 결과적으로 수정된 구조를 통해서 WER 이 좋아짐을 확인함 (7.28%→7.12%)

## 3.2 Micro-Architecture Design

### 3.2.1 Unified Activations

- Conformer은 대부분의 block에서 Swish activation을 사용하지만, Conv module 에서는 GLU 으로 전환됨
- 이러한 계층적인 구조는 너무 복잡하고 불필요하기 때문에 본 연구에서는 GLU activation을 Swish으로 변경하여서 activation function을 통일해주는 과정을 거쳤음
- 이러한 변화는 성능향상으로 이어지지는 못했지만 전체적인 모델 아키텍쳐를 simplify 해줬음

### 3.2.2 Simplified layer normalizations

- conformer VS squeezeformer
    
    ![Figure 4: (Left) Back-to-back preLN and postLN at the boundary of the blocks. (Right) The preLN can be replaced with the learned scaling that readjusts the magnitude of the activation that goes into the subsequent module.](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/1b674492-f670-4eb5-ae1d-c61c3a2d2d25){: width="80%" height="80%"}{: .center}
    
    Figure 4: (Left) Back-to-back preLN and postLN at the boundary of the blocks. (Right) The preLN can be replaced with the learned scaling that readjusts the magnitude of the activation that goes into the subsequent module.
    
    - 원래의 conformer은 redundant layer norm을 fig4 와 같이 적용했음.
        - 즉 post와 pre Layernorm이 모두 적용된 것을 볼 수 있음
            - postLN은 residual block 사이에, preLN은 residual connection 안에 적용
            - 이건 preLN이 training을 보다 안정화 시켜주고, postLN은 성능을 향상시켜준다는 가정하에 진행된 것
        - 하지만 이렇게 연속적인(back-to-back) 연산은 global reduction 연산으로 인해서 계산량이 많아질 수 있음.
        - 그런데 연구진들은 단순하게 pre LN이나 post LN을 제거하는 것은 training 불안정과 convergence 실패로 이어진다는 것을 확인했음
            - 이러한 현상이 발생한 원인은 전형적으로 훈련된 conformer model의 preLN과 postLN은 학습가능한 scale 변수들의 norms의 magnitude 가 다르기 때문
            - 특히 연구진들은 preLN이 큰 값으로 input signal을 scale down해서 skip connection에 좀 더 가중치를 준다는 것을 확인했기에 pre LN을 대체할 때 scaling layer을 사용했음.

### 3.2.3 Depthwise Separable Subsampling

- 단일 모듈을 수정하는 것이 간과되어서는 안되는게 conformer-ctc-m의 경우 30초의 입력이 들어왔을때 subsampling block에서 전체 FlOPs의 28%가 발생되었음
- 이는 subsampling layer가 2개의 vanilla convolution operation을 stride 2로 설정하여 사용하기 때문인데, 이를 해소하기 위해서 본 연구에서는 두번째 convolution operation을 depthwise seperable convolution으로 바꿨음
- 첫번째 convolution과 같은 경우에는 input 차원이 1일때 depthwise conv와 동일하기 때문에 그대로 놔둠
- 그 결과 WER의 변화없이 baseline FLOPs 를 22%정도 감소시킬 수 있었음.
    - test-clean WER에서는 0.06%의 WER 향상도 보임

# 4 Results

- 실험 결과 WER improvement와 함께 training stability를 보장함.
    


<br/>
<br/>
<div id="disqus_thread"></div>
<script>
    (function() { // DON'T EDIT BELOW THIS LINE
    var d = document, s = d.createElement('script');
    s.src = 'https://https-jisoo0-0-github-io.disqus.com/embed.js';
    s.setAttribute('data-timestamp', +new Date());
    (d.head || d.body).appendChild(s);
    })();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/