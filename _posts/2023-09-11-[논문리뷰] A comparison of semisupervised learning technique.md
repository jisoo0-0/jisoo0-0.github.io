---
layout: post
categories: 논문리뷰
tags: semi-supervised
comments: true
math: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] A comparison of semi-supervised learning techniques for streaming ASR at scale"
---


**논문 및 사진 출처**
>Peyser, Cal, et al. "A Comparison of Semi-Supervised Learning Techniques for Streaming ASR at Scale." ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023.    


# Abstract

- unparied text와 audio injection은 ASR 성능을 향상시키는 중요한 방법 중하나였음
- 하지만, 이러한 방법을 사용해서 ASR system을 향상시키기 위해서는 guidance에 대한 연구가 많이 없음
- 본 연구에서는 SOTA semi-supervised mthods를 unpaired text and audio와 함께 controlled setting에서 비교하고자 함
- 본 연구에서 제안한 setting을 통해서 성능 개선을 보임

# 1 Introduction

- speech recognition의 문제를 해결하기 위한 방법으로 semi-supervised learning이 제안됨
- semi-supervised training scheme에서는 unpaired speech 혹은 text examples이 supervised dataset에 보충제역할로 제공되어서 더 좋은 language / acoustic coverage를 가질 수 있도록 함
- audio와 text을 사용한 semi-supervised learning은 ASR task에서 아주 좋은 성능을 보였음.
- 하지만, 제안된 방법들은 주로 industrial scale에서 제공된 데이터셋보다 더 작고, supervised data가 거의 없는 사례들에 초점이 맞추어져 있음
- 또한, 현재 mobile phone이나 streaming prediction이 가능해야하는 현실 세계에서의 application과는 괴리가 있는게 모델의 크기가 엄청 큼.
- 본 연구에서는 몇개의 leading semi-supervised methods을 비교하는데 제한된 setting에서 비교를 진행함.
    - 여기서 말하는 state-of art semi-supervised methods는 TTS augmentations, JOIST, BEST-RQ를 의미함.
    - 이전 연구들과는 다르게, 본 연구에서는 semi-supervised methods을 conformer에 적용했음
        - 여기서 conformer은 160M-streaming conformer을 의미하고, 엄청나게 큰 supervised corpus에 대해서 미리 학습된 모델임
- decoder의 computational load와 lattice density에 대해서 improvements을 보였음.

# 2 Related work

## 2.1 Text injection

- Unsupervised text injection in ASR은 LM에서 fusion으로 주로 수행되어 왔음.
    - inference time / training time 모두에서
    - 이러한 방법은 paired data에 훈련된 acoustic model과 unpaired text에 훈련된 LM의 모델 파라미터를 명시적으로 분리하게 하는 것.
    - 이러한 improvement가 있었지만, inference time에서 additional LM param에 대한 cost를 요하게 되었음
- fusion 을 사용하지 않는 대안이 모색되기는 했음.
    - unsupervised text을 사용해서 acoustic model을 직접 훈련하는 방법
    - 여기에서 자주 사용되었던 방법중 하나는 synthesized audio를 통해서 수도 라벨을 생성하는 것임.
- unpaired text injection을 위한 또다른 방법은 TTsS 수도 레이블을 생성하지 않은 채 ASR encoder을 학습하는 것임. 하지만 이러한 작업들은 ASR encoder가 텍스트나 오디오 둘중하나를 나타내도록 훈련이 되고, unpairedtext가 audio와 유사하게 처리되는 것을 요했음

## 2.2 Audio injection

- Unsupervised audio injection에 대한 연구는 활발하게 진행되어 왔음.
- constrastive loss를 사용하거나 audio input을 양자화 하는 방법이 있음

# 3 Methods

## 3.1 Architecture

- E_c : causal encoder
    - 오른쪽 범위의 context를 입력받지 않은채 오디오 feature을 처리
- E_nc : non-causal encoder
    - Ec의 output과 오른쪽 범위의 900ms 의 입력 값 사용
- D_c : causal decoder
    - inference때 해당 decoder은 immediate prdiction을 산출해냄
- D_nc : non-causal decoder
    - inference 때 해당 decoder은 short latency로 causal decoder의 prediction을 수정하는데에 사용됨
- 이전 연구들과 다르게, 본 연구에서는 audio 와 text 중 하나의 representation을 consume하도록 했음.
    - 이를 위해서 JOIST를 따랐는데, E_c가 입력
    - 두개의 neural frontedns를 선택하는데 하나는 audio feature들을 위한거고 나머지 하나는 text feature을 위한 것임.
    - JOIST에서 우리는 text fronted output을 upsample해서 text와 audio의 길이가 대략적으로 비슷하게 될 수 있게 맞춰줬음

## 3.2 Tasks

- causal ASR에서 x는 audio frontend에 전달되어서 Ec로 encoding되고, Dc로 decoding됨.
- noncausal SR에서는 E_nc, D_nc를 사용해서 훈련
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/79710b53-dc62-41af-8bb6-a83ace8bb0ce)
    
    - 파란색 선 : causal
    - 초록색 선 : non-causal
- 본 연구에서의 모델은 RNN-T loss를 사용해서 end-to-end로 training됨

### 3.2.1 TTS augmentation

- frozen parameter을 가지고 pre-trained된 TTS system을 사용해서, 본 연구의 저자들은 audio clip ~\hat{x} 을 생성해 냄. ~\hat{x}는 unsupervised text segment y 에 대응되는 audio clip임.
- 본 연구에서는 (~\hat{x},y}를 supervised audio-text pair으로 treat해서 causal 과 non-causal ASR task들을 훈련함.
- 이는 figure 1에서 dotted blue와 dotted green으로 표현되어 있음.
- 저자들은 합리적인 학습 속도를 달성하기 위해서, TTS system 이 input word-pieces를 raw audio롤 convert 하는 것이 아니라, acoustic features의 sequence으로 convert하는 것이 중요하다고 주장했음.
    - sequence of acoustic feature은 audio frontend으로부터 consumed됨
    - 이는 decoder이 autoregressive하게 작동되기 때문에, audio sequence 의 길이가 training speed에 영향을 미치게 되기 때문.

### 3.2.2 JOIST

- masked unpaired text examples을 text frontend에 입력시켜주는데, text frontend는 learnd projection으로 구성되어 있음.
- 결과는 오디오와 같이 E_C, D_C 혹은 E_nc, D_nc로 차례대로 전달되고 RNN-T손실을 사용해서 원본 text sequence와 비교하게 됨
- 본 연구의 저자들은 JOIST가 text token과 대조적으로 y의 음소 representation들을 consume하는 것이 중요하다는 것을 참고논문을 통해서 주장했음.
- masking 전에 text를 처리하는 text-to-음소들 lookup을 포함함.
- JOIST loss는 standard한 word-piece representation에 대해서 작동함
    - JOIST loss는 masked 된 음소 시퀀스로부터 word pieces을 생성하는 것으로부터 학습됨

### 3.2.3 BESt-RQ

- BEST-RQ 이후에 audio injection을 modeling했음
- Audio feature들은 masked 되고 fronted를 사용해서 process됨
- 이후, causal 혹은 non-causal encoder을 통해서 encoding함.
- 추가적으로 audio feature들은 randomly init된 projection에 의해서 process되고, fixed codebook에 있는 가장 가까운 entry으로부터 반올림해서 이산화됨
- 이후, encoder은 masked 된 region에 대해서 quantized target을 예측하도록 학습되고 이는 figure 1에서 빨간색 dashed line으로 표현되어 있음.

# Conclusion

- 해당 포스트에는 실험결과가 생략되어있지만, 실험 결과를 통해서 본 연구는 semi-supervised learning을 ASR system에 적용할 수 있는 insight를 제공하고, semi-supervised technique에 따라 ASR performance가 어떻게 달라지는지 보였음.(특히 challenging 한 acoustic scene에서)
- 온디바이스 시스템의 구성요소로 ASR이 적용되었을 때, decoding state와 lattice richness를 측정함



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