---
layout: post
categories: 논문리뷰
tags: targetlearning, multimodal, self-supervised
comments: true
math: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] Efficient self-supervised learning with contextualized target representations for vision, speech and language"
---


**논문 및 사진 출처**
>Baevski, Alexei, et al. "Efficient self-supervised learning with contextualized target representations for vision, speech and language." International Conference on Machine Learning. PMLR, 2023.    


# Abstract

- 현재의 self-supervised learning 알고리즘은 modality-specific 하거나 computational resource들을 많이 요구한다. 이러한 문제점을 해결하기 위해서, data2vec의 training efficiency를 증가하고자한다.
    - masked token을 인코딩하지않고 빠른 conv decoder을 사용하며 teacher 모델의 representation을 학습하고자 함

# 1 Introduction

- self-supervised learning은 주로 single modality를 기준으로 설계되었음
- 최근에 unified model architecture이 등장하고 다른 modality들에 같이 사용될 수 있는 loss fucntion이 등장하기는 함
- 현재의 self-supervised model은 efficient하지는 않음.
- 본 연구에서는 data2vec 2.0을 사용해서 self-supervised learning의 효율을 증가하고자 하며, contextualized target prediction을 얻고자 함
    - efficient data encoding, fast conv decoder, 각 sample의 다양한 masked version들에 대한 target representation들 재사용
    - 본 연구에서 제안된 알고리즘은 각각의 modality에 대해서 같은 learning objective을 사용하지만, 인풋 모달리티에 따라서 다른 feature encoder을 사용함
    - 아래 그림과 같은 training을 진행함
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/3339c692-5f8e-45f4-8075-5688467cd35b)
        
    - teacher model을 통해서  latent contextulized representation 생성
        - unmasked training samle들은 student model으로부터 regressed 됨
        - target contextualization은 전체적인 sample에 대한 information을 capture할 수 있도록 돕는다. (ex. 같은 단어가 다른 문장에서 다른 의미를 가질 수 있는 것처럼)
        - LOSS
            
            ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/d26d0783-a110-462e-a1b1-9db04862f71a){: width="30%" height="30%"}{: .center}
            

# 2 Related work

- pass

# 3 Method

- 3-1 contextulaized target representation을 예측하는데에 사용되는 shared technique 설명
- 3-2 encoder decoder 설명
    - mased auto encoder들과 비슷하게, 본 연구에서는 non-masked portion 들만 encoding하고 decoder가 target representation들의 masked portion 들을 예측하게함
    - 대신 Transformer-based decdoer대신에 더 크기가 작은 conv decoder을 사용해서 더 빠르게 학습하고자함
- 3-3 contextualized target representation을 생성할 때 사용되는 계산양을 감소시키기 위해서 training sample에 대한 multiple한 masked version들을 재사용함
- 3-4 random masking이나 block masking 대신에 inverse masking 전략을 사용
    - inverse block masking 전략은 sample의 연속성을 보장해주어서 좀더 많은 문맥 정보를 줄 수 있다고 저자는 주장함

## 3.1 Contextualized target prediction

- raw input data에대해서 local window를 reconstruct하거나 discreate 한 representation을 predicting 하는 것 대신, 전체 input sample으로부터 teacher network에 대한 representation을 예측하고자 함
- contextualized target을 self-attention mechanism을 통해서 생성됨
    - training target은 모든 sample에 대한 weighted sum임

### 3.1.1 Target representation and learning objective

- Training target은 teacher model의 top K개의 FFN block을 averaging한 것에 기반함.
- averaging 전에는 activation들이 정규화됨 (instance norm사용)
- Student network에 대한 training task는 이러한 target을 regress하는 것임
    - sample의 masked version 사용

### 3.1.2 Teacher weights

- Student encoder weight에 대해서 Exponentially moving average한 것.

### 3.1.3 Learning objective

- teacher network y에 대해서 target representation에 L2 loss 적용

## 3.2 Model architecture

- data2vcec과 비슷하게 modality specific feature encoder을 사용하고, transformer architecture 을 사용함.
- CV task에서는 patch mapping을 사용하고 , speech task에서는 multi layer conv를 사용하고 text task에서는 byte-pair encoding을 통해서 학습된 embedding을 사용함

### 3.2.1 Asymmetric Encoder / Decoder architecture

- teacher model을 사용해서 모든 unmasked training sample을 encoding해서 training target을 생성함
- sample의 일부를 마스킹하고, student encoder을 통해서 embedding진행
- efficiency를 위해서 unmasked patch 혹은 time -step만을 encoding함.
- student encoder의 output은 마스킹된 부분에 대한 fixed representation과 함께 merged되어서 decoder network으로 전달됨.
- masked token의 representation을 위해서 랜덤 가우시안 노이즈를 사용하는 것이 효율적이라는 것을 발견하였음.
- decoder entwork은 teacher network의 time-step들에 대한 contextualized target을 reconstruct함

### 3.2.2 Convolutional decoder network

- conv + layer norm + GELU activation + residual connection 사용
- speech와 text와 같은 sequencial data같은 경우에는 1D conv를 사용햇고, 이미지와 같은 경우에는 2D conv를 사용했음.
- 각 모달리티에 대해서 커널 사이즈랑 레이어 개수 튜닝 진행

## 3.3 Multi-mask training

- 기존 data2vec의 techer-student setup에서의 불이익은 sample에 대한 process를 두번 진행해야한다는 것이다.
    - 한번은 teacher model을 통해서 target 을 얻음
    - 나머지 한번은 student model을 통해서 prediction 을 얻음
- 더 나아가서 teacher model에 대해서 activation을 계산하는 것이 조금 비효율적임.
    - 이는 teacher model이 모든 unmaksed input을 처리해야하기 때문임
- teacher model computation cost을 완화하기 위해서 각기 다른 masked version에 대한 teacher representation을 다시 사용하기로 했음
    - training sample의 M 개의 다른 masked version 에 대해서 target representation과 비교
- feature encoder output을 공유함.
    - 각기다른 masked version에 대해서 이를 공유함으로서 계산에 대한 중복 방지- > 계산량 감소

## 3.4 Inverse block masking

- MAE(masked auto encoder) style sampling encoding은 efficiency를 향상시켜주지만, masked time-step 들에 대한 activation에 있는 정보들을 저장하지 못하게 함.
    - random masking은 semantic한 representation이 생성되기 힘듦
    - block masking은 연속적이면서 긴 portion들이 unmasking되어 있는 상태를 개런티해주지는 않음
- 본 연구의 목표는 student model이 semantically하게 풍부한 representation들을 형성하는 것임.
    - local한 region에 대해서 진행하는 것을 방지하고자 한듯
- Inverse block masking을 통해서 위의 목표를 성취함
    - masking할 patch를 정하는 대신, 어떤 patch가 보존되어야할지를 결정함.
        - block-wise 방식으로 진행됨
        - size of the block은 number fo the patches / time-steps B와 같음
    - 각 block의 starting point는 보존되게 했고, block의 너비가 patch나 time-step의 너비와 같아질때까지 증가시킴.
    - starting point에 대한 일정 개수를 sampling함.
        - $L \times \frac{(1-R) + A}{B}$
            
            L : time step 혹은 patch의 개수
            
            R : masking ratio
            
            A : masking ratio 조정 hyper-param
            
    - block이 overlap될 수 있도록 함

# 5 Conclusion and future work

- 효과적이고 general 한 pre-training 기술 제안
    - 다른 modality에서도 적용 가능
- data2vec은 training speed가 엄청 빠름.
- 요약하면 다양한 모달리티에 적용가능한 pre-training 기술을 제안했는데, conv decoder을 사용해서 속도도 빠르게 하고 새로운 masking을 제안하면서 sementic한 representation을 효율적으로 학습할 수 있게 했다는 거에 있는듯.



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