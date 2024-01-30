---
layout: post
categories: 논문리뷰
tags: audio, voice_cloning
comments: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] Self supervised learning for robust voice cloning"
---

출처
> Klapsas, Konstantinos, et al. "Self supervised learning for robust voice cloning." *arXiv preprint arXiv:2204.03421* (2022).


Copyright of figures and other materials in the paper belongs to original authors.



# Abstract

- Voice cloning은 견고하고 informative한  feature들을 필요로 함
    - 이는 이러한 feature들을 통해서 unseen speaker의 목소리를 효과적으로 복사하기 위함임
- 본 연구에서는 BYOL(Bootstrap Your Own Latent) 방법을 통해 self supervised 방식으로 학습된 feature들을 사용함
    - BYOL은 vanilla 알고리즘에 specific한 오디오 증폭 방식이 적용되었을 때 high quality speech representation을 생성할 수 있음
    - 더 나아가서 학습 절차에 있는 증폭을 확장해서 feature들이 speaker의 identity를 더 잘 포착하도록 함과 동시에 noise나 acoustic condition(음질)에 견고할 수 있도록 함
    - learned feature은 pre trained utterance level embedding으로 사용하였으며 , Non Attentive Tacotron 기반의 구조의 input으로 사용하였음
        - 이는 multi speaker speech 합성을 추가적인 speaker feature 없이도 가능하게 했음

# 1 Introduction

- acoustic model에 learnable speaker embedding을 제공해 주는 것은 multi speaker speech 합성을 보다 더 용이하게 했음
- training set에 포함하지 않은 speaker의 목소리를 복제하는 것은 unseen data에 일반화할 수 있는 representation을 요함

## 1.1 Related Work

- speaker embedding은 single model상황에서 speaker들이 다양할 때 해당 speaker들을 구분해 낼 수 있는 적합한 representation임
- multispeaker model을 새로운 speaker에 적용하는 다양한 방법이 존재함
    - fine tuning, target speaker의 data를 base model을 training 시킬 때 사용하기, 등
- data가 부족한 상황에서는 speaker encoder을 사용해서 audio 으로부터 speaker embedding을 바로 predict할 수 있는 방안이 사용될 수 있음
- 다양한 연구들이 speaker 의 representation을 일반화하기 위해서 진행되어 왔음
    - x-vectors for speaker recognition , d-vectors for speaker verification
        - 이 두가지 기법은 speaker의 embedding의 길이가 고정되어 있었고 화자에 대한 정보가 하나의 벡터에 담기기 때문에 정보가 불충분하게 반영된다는 단점이 있음
    - speaker의 embedding을 fix시키지 않고, audio으로부터 가변적인 길이의 embedding을 추출해낸 연구도 있었음
    - self supervised learning은 machine learning기법으로 model이 스스로 학습하는 기법임
        - 데이터를 일부를 활용하여 supervisory signal을 생성해서 학습
    - pre training method, constrative approach, .. 등이 사용되었음
    - BYOL
        - audio data에 적용되어서 general purpose embedding을 생성하고자 했음
        - augmentation기법 사용됨
    

## 1.2 Our contributions

- unlabeled 된 dataset에서도 사용될 수 있는 voice cloning algorithm을 제안함
    - 다수 speaker 상황
- training set의 일부만 사용했음에도 불구하고 d-vector을 사용한 베이스라인의 모델과 비슷한 voice cloning 성능을 보임
- additional 한 augmentation을 통해서 learned self supervised representation을 생성하였으며 이는 robust한 voice cloning을 가능하게 하였음

# 2 Method

## 2.1 Overview

- 본 연구는 non attentive Tacotron TTS 구조에 기반하여 LPCNet Vocoder을 위한 feature을 생성해 내려고 하고,  pre trained 된 self supervised feature에 기반함
- 본 연구에서 사용된 pre training 방법은 BYOL 알고리즘임
    - 해당 알고리즘은 self supervised learning method으로 meaningful representation을 효과적으로 추출해 냄
    - original 알고리즘을 통해 audio representation을 학습한 BYOL-A가 존재했음
    - 해당 알고리즘이 labeled dataset없이도 speaker의 identification을 구별해 낼 수 있었기 때문에 본 연구에서는 음성 복제를 위해서 speaker의 embedding으로 audio representation을 TTS system에 적용함
    

- Non Attentive Tacotron?

> Shen, Jonathan, et al. "Non-attentive tacotron: Robust and controllable neural tts synthesis including unsupervised duration modeling." *arXiv preprint arXiv:2010.04301* (2020).


- Tacotron은 attention을 활용한 encoder decoder구조임
- Encoder 부분에서는 input phoneme을 통해 hidden feature representation을 생성함

## 2.2 BYOL for audio

- BYOL은 동시에 2개의 NN(Neural Network)을 훈련함
    - online과 target network
- 두개의 network은 동일한 구조를 사용하지만 다른 가중치를 사용함.
    
    $$
    Online Weight=θ , TargetWeight=ξ
    $$
    
    - 두개의 network은 동일하게 **representation encoder f**, **projector g**으로 구성됨
    - 추가적으로 online network은 추가적인 **prediction module q**가 있음
- audio 의 두개의 augmented view 를 아래 식과 같이 생성함
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/1bdb44ca-b2fc-4314-a748-65754e887f20){: width="20%" height="20%"}{: .center}
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/e6243e06-1593-4af4-a936-e4cebc09b607){: width="20%" height="20%"}{: .center}
    
    - t, t` : audio augmentation,  audio augmentation  으로부터 샘플링된 것
- online network은 representation과 projection을 출력함
    - inference에서는 representation만 사용됨
- target network은 두번째 augmented view를 사용해서 target projection을 출력함
- 전체 loss식은 아래와 같음
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/a8ef30d2-4d6b-4b12-a5a7-b05c4a5dbfb3){: width="50%" height="50%"}{: .center}
    
    - 해석을 해보자면.. q가 online network의 추가적인 prediction module이고, z가 projection 정보이니까, target에 대한 projection 정보를 prediction module에 입력한 값과, target projection의 값간의 차이를 loss로 둔 것
- augmentation의 측면에서도 loss가 동일하게 적용되기 위해서 u` augmentation이 online network에 , u는 target network에 fed되었음. 따라서 final loss는 아래와 같음
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/7d048915-8a80-4c3e-9eaf-4d6290fd826f){: width="30%" height="30%"}{: .center}
    
- target network의 파라미터는 online parmeter의 지수 가중 평균을 통해서 업데이트 됨 식은 아래와 같음
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/a00c871c-c1b8-4758-a955-ae8b338dfb09){: width="30%" height="30%"}{: .center}
    

## 2.3 BYOL-A Augmentations

### 2.3.1 Pre and Post Norm

- pre, post 모두 다 평균과 표준편차를 기반으로 적용되었음

### 2.3.2 Mixup

- Mixup block은 랜덤하게 선택된 과거의 input과 current input을 섞는 것
- 과거 input은 background sound으로 사용되어서 network이 foreground acoustic event의 representation만을 학습할 수 있도록 도움
- acoustic feature이 log scaled되었기 때문에, 이들은 linear한 sacle으로 변환되었고 그 다음에 mixup이 적용되었음
- BYOL-A에서 mixup block이 사용된 이유는 foreground와 background를 구별하기 위해서인데, 본 연구에서는 speaker간의 구별가능한 feature을 더 학습할 수 있도록 사용됨
    - 즉 본 연구에서는 다른speaker의 데이터로부터 샘플링된 과거 input이 포함될 수 있다느 ㄴ것

### 2.3.3 RRC
- Random size crop은 image augmentation 기술인데, audio task에서는 Mel spectrogram에 적용되었음
    - pitch shifting 이나 time stretching에 사용될 수 있음
- RRC는 log mel spectrogram에 random하게 crop한 sample을 포함함
- frequence bin F와 time frame T가 주어졌을 때 아래와 같이 crop area 의 크기가 랜덤하게 샘플링 됨
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/957e785e-0698-4eba-aee4-75ab26813793){: width="45%" height="45%"}{: .center}
    
    - [바닥함수](https://ko.wikipedia.org/wiki/%EB%B0%94%EB%8B%A5_%ED%95%A8%EC%88%98%EC%99%80_%EC%B2%9C%EC%9E%A5_%ED%95%A8%EC%88%98), ⌈…⌉[천장함수](https://ko.wikipedia.org/wiki/%EB%B0%94%EB%8B%A5_%ED%95%A8%EC%88%98%EC%99%80_%EC%B2%9C%EC%9E%A5_%ED%95%A8%EC%88%98)
    - Fc : frequencey frame의 개수
    - Tc: Time frame의 개수
    - h1, h2, w1, w2는 scaling 범위

### 2.3.4 Gaussian Noise

- 가우시안 증강 블록은 training data와 정규 분포로부터 샘플링된 noise간의 interpolate을 해줌
- 해당 증강 기법의 목표는 mixup augmentation과 비슷함
- 가우시안 노이즈는 N(0,0.04)으로 설정되어 log domain에 더해짐
- 이는 mixup augmentation에서의 log exp trick과 동일함

## 2.4 Additional Augmentations for Robust Voice Cloning

### 2.4.1 Prosodic Augmentations

- plain RC와 Mixup이 cloning을 할 때 충분했는데, direct pitch shifting과 duration scaling 을 waveform에 바로 적용하는 것이 더 좋은 성능을 보임을 확인함
- prosodic variatin은 speaker의 identity에 영향을 주면 안되기 때문에, 이를 사용해서 augmentation을 하는 것이 self supervised 방식에서 speaker identity에 모델이 보다 focus할 수 있도록 도울수 있음
- pitch shifting과 duration scaling은 Praat Tolkit을 통해서 적용되었으며, 너무 많이 변화를 주는 것은 speaker의 similairty 에 영향을 주기 때문에 적은 변화만 가지고 증폭이 진행되었음

### 2.4.2 External Noise

- BYOL-a feature들을 학습하는 것이 일반적인 목표이기 때문에 각 발화의 acoustic condition이 representation에 present되어 있을 것이라고 예상함
- 발화자의identity가 노이즈가 있더라도 invariant하기 때문에 dataset속에 존재할 수도 있지만 이게 본 task에서 적합하지 않음
- self supervised feature을 acoustic conditon에 보다 robust할 수 있도록 생성하기 위해서 backgorund noise dataset을 사용했음
- 제안된 augmentation은 랜덤하게 선택된 nosie와SNR을 waveform에 적용함

# 3 Experiments and Results

## 3.2 Objective Evaluation

- s2t-same의 선행연구에서 사용된 metric을 사용함
- synthesized audio가 동일한 speaker의 groundtruth audio와 얼마나 유사한지 측정함
- 결과(낮으면 낮을수록 좋음)
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/0e7a7bac-aeb6-4f57-b118-be3622b8a81a){: width="60%" height="60%"}{: .center}
    
    - Method에는 음성 증강 방법이 적혀 있음
    - de vocetor pretrained with VCTK보다는 저자들이 제안한 모델의 성능이 더 좋음
    - de vocetor pretrained with VoxCeleb2가 가장 좋은 성능을 보이는데, 이는 해당 모델이 더 많은 dataset을 기반으로 pre train되었기 때문에 더 좋은 일반화 성능을 보이는게 아닌가 함

## 3.3 Subjective Evaluation

- pass

# 4 Conclusion

- unlabeled dataset에 pre trained 된 self supervised feature을 기반으로  voice cloning 구조를 제안함
- dataset의 일부만 사용하고도 baseline과 비슷한 성능을 보임
- cloning의 성느을 향상시키기 위해서 augmentation기법을 사용하였고, cloning 성능과 질을 올려서 target 발화의 노이즈에 보다 더 견고한 결과물을 보일 수 있게 했음


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