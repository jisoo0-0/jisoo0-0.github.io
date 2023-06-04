---
layout: post
categories: 개념정리
tags: audio,ASR
comments: true
disqus: 'https-jisoo0-0-github-io' 
title: "[개념정리] ASR System / RASR / RETURNN / Comformer"
---

#0 ASR System

##0.1 ASR System이란?

- **ASR(Automatic Speech Recognition) system이란?**
    - 음성 인식 시스템은 여러 구조가 있을 수 있지만, 고전적인 구조는 크게 4가지 모듈로 구성됨
        - Pre-processing : Filter를 사용해 Signal to Noise Ratio를 줄여줌
        - Feature Extraction : 음성 신호를 시간 단위로 잘라 주파수 특징을 추출 (MFCC 등 벡터), Acoustic Model의 Input이 됨
        - Classification(Acoustic Model) : 추출된 Acoustic Feature(특징 vector)를 이용해 발화된 음소열을 판별하며 주로 통계적인 Model 사용 (GMM-HMM)
        - Language Model : AM에서 출력된 음소열을 문자열/단어로 변환 (N-gram)
- **Hybrid ASR System**
    - Deep Learning 기반 Algorithm의 성능이 향상되어, ASR에도 고전적인 GMM-HMM 기반 Acoustic Model보다 DNN을 사용한 음성모델의 성능이 좋아짐
    - GMM의 역할을 DNN이 대체하는 DNN-HMM 등이 Hybrid ASR Model
        - GMM(Gaussian Mixture Model)은 각 state의 distribution을 여러 Gaussian을 합성해서 modeling하고 이 확률을 제공하는데 현실에서는 이 가정이 항상 정확하지는 않음
            
            → DNN은 Nonlinear Function로 학습이 가능하여 대체 가능 (➕)
            
            → 하지만 Layer가 많아지면 학습이 어려움 (➖)
            
        - DNN의 예시로는 MLP, RNN 계열(LSTM), Transformer 등이 있음
- 본 연구에서 제안하는 Hybrid Model은 Conformer base의 Acoustic Model
    - 특징 : Local Feature를 잘 catch하는 CNN과 Global feature를 잘 잡는 Transformer를 합친 구조로 Conformer Block 내부에 Attention Module과 Convolution Module이 같이 존재함
        
        ![음향모델 종류 및 DNN에서 사용되는 여러 Model의 예시](http://drive.google.com/uc?export=view&id=1yCLL9xRQLouVmLls_pX6A77RzqjJaWwK){: width="80%" height="80%"}{: .center}
        
        음향모델 종류 및 DNN에서 사용되는 여러 Model의 예시
        

# 1 RASR / RETURNN / Comformer

## 1.1 RASR

- 해당 모델은 본 연구에서 사용된 모델이기에 간단한 리뷰를 진행함

> Rybach, David, et al. "Rasr-the rwth aachen university open source speech recognition toolkit." *Proc. ieee automatic speech recognition and understanding workshop*. 2011.
> 

### 1.1.0 Abstract

- **<mark>RASR은 speech recogniton 오픈 소스 toolkit임</mark>**
- 논문 발표 시점(2011년도)를 기준으로 음성 모델 training과 decoding에 SOTA 성능을 보임

### 1.1.1 Introduction

- pass

### 1.1.2 Signal Analysis

- Flow networks
    - flow module은 일반적인 data processing을 위한 framework을 제공함
    - flow network은 acoustic한 feature을 계산하고 dynamic 한 time alignment을 생성하고 process하기 위해서 사용되었음
        - ex. acoustic feature을 HMM state으로 mapping하기
- Acoustic features
    - Flow netowrk의 basic node은 audio file으로부터 waveform을 읽고, FFT를 계산하고, vector opreation하고, 다양한 signal normalization을 시행하는 것임
    - 툴킷에 포함되어 있는 network은 MFCC feature을 계산하는 것, feature을 voicing하는 것을 포함하고 있음.
    - Temporal한 context는 acoustic feature의 미분값을 사용하거나 LDA transformation을 통해서 통합될 수 있음

### 1.1.3 Acoustic Modeling

- Acoustic model은 transition, emission, pronunciation model으로 구성됨
    - pronunciation model은 vocab에 있는 각 단어에 대해서 각 단어가 발생할 확률과 발음될 목록을 제공함
        - pronunciation은 context에 기반한 phonemes에 따라서 모델링 됨
            - 현재 버전에서는 context가 triphones에 한정됨
        - left-to-right HMM topologies 가 support되고, 각각은 context에 기반한 phoneme을 represent하게 됨
            - silence을 제외하고는 모든 HMM은 같은 개수의 state을 가짐
    - transition model은 loop, forward, skip transition을 시해암
        - 현재 버전에서는 global한 transition mdoel을 제공하는데, 이 모델은 silence state만을 구별할 수 있음
    - HMM state의 emission probability는 가우시안 mixture model을 기반으로 represent 됨
        - 기본적으로 globally하게 pooled 된 variance가 사용됨
- 본 연구에서 제공하는 툴킷은 HTK acoustic model을 RASR model으로 convert 할 수 있는 tool도 제공하지만 HTK 모델에 있는 모든 파라미터가 사용되는 것은 아님
    - HTK는 또다른 acoustic toolkit임
- State Tying : Phonetic Decision Trees
    - RASR은 Classification And Regression Tree(CART) 을 phonetic decision tree을 위해서 train하는 tool을 포함함
    - CART training은 유연한 음성 의사 결정 트리 기반의 tying을 제공함
        - 예를 들어서 각 언어마다 separtation의 정도에 따라서 성능이 다름
        - 추가적으로 CART software은 시스템의 조합의 subsequent을 위해서 랜덤하게 여러개의 acoustic model을 생성하는 것을 support함

### 1.1.4 Confidence Scores

- pass

### 1.1.5 Speacker Normalization and Adaptation

- **RASR 은 speaker normalization과 adaptation을 위한 몇몇의 방안을 support 하기도 함**
    - Vocal tract length normalization(VTLN)
        - MFCC filter bank에서 parametric linear warping을 적용한 것
        - 파라미터는 maximum 가능성을 기반으로 estimated됨
        - Maximum likelihood linear regression(MLLR)
            - Regression class tree approach는 regression classes을 조정하기 위해서 사용됨

### 1.1.6 Acoustic Model Training

- **RASR 는 가우시안 mixture model을 추정하는 tool을 support 함**
    - standard maximum likelihood training과 discriminative training을 MPE criterion를 이용한 training 을 통한 model

### 1.1.7 Language Modeling

- **해당 toolkit은 language model에 관한 estimation을 위한 tool을 support하지는 않음**
- **하지만, decoder에서는 N-gram language model 을 APPA formet으로 제공해주기는함**

### 1.1.8 Decoder

- **History Conditioned Lexical Tree(HCLT) search을 제공하는 decoder을 RASR toolkit에서는 제공함**
    - HCLT search 는 one-pass dynamic programming 알고리즘으로 pre-complied 된 lexical prefix tree을 pronunciation dictionary으로 사용함
        - <mark>**search space이 언어 모델의 integrating part을 통해 동적으로 구성되기 때문에 decoder이 아주 큰 vocab을 처리할 수 있고 , 복잡한 언어 모델을 다룰 수 있게 됨**</mark>
- Decoder은 word graphs(lattices)을 생성할 수 있음
    - lattices은 word boundary들에 상응하는 word sequence의 alternative한 set의 compact한 representation임
    - lattices는 processing step의 후반에 사용될 수 있음

### 1.1.9 Lattice processing

- Lattice processing tool은 decoder으로부터 생성된 word graph의 post-processing 으로 사용될 수 있음
- 본 framework에서 사용된 주요한 방법은 아래와 같음
    - Confusion network construction
    - Confusion network decoding
    - Lattice와 Confusion network 기반의 system을 조합한 것
    - n-best list generation
    - word confidence score 예측
- 위의 방법들은 basic한 operation과 같이 사용되어 data processing network을 형성할 수 있음
    - lattice pruning, file operation, format conversation

## 1.2 RETURNN

- 해당 모델은 본 연구에서 사용된 모델이기에 간단한 리뷰를 진행함

> Doetsch, Patrick, et al. "RETURNN: The RWTH extensible training framework for universal recurrent neural networks." *2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE, 2017.
> 

### 1.2.0 Abstract

- The software **allows to train state-of-the-art deep bidirectional long short-term memory (LSTM) models** on both one dimensional data like speech or two dimensional data like handwritten text and was used to develop successful submission systems in several evaluation campaigns.
    - 해당 software는 speech 학습이나 손글씨 text 데이터를 Bi LSTM을 통해서 학습할 때 SOTA 성능을 보임

### 1.2.1 Introduction

- LSTM은 ASR을 포함하는 sequential 한 learning task에서 아주 dominent 하게 사용되고 있음
- Theano 가 RNN학습을 잘 할 수 있는 패키지를 제공했지만, large scale task를 위한 solution은 제공하지 않았음.
- RETURNN 은 Theano을 additional layer으로 사용함

### 1.2.2 Related Work

- Theano 는 python 기반 tensor expression이 가능한 frame work임
- Keras는 Theano의 highlevel framework으로, RETURNN과 가장 유사한 software package임
    - Tensorflow를 backend으로 서포트 해주고 있기도 함
- Tensorflow는 google이 제공해주는 머신러닝 오픈소스 패키지
- **RASR 는 Task specific한 software package으로 speech reconition system을 위한 것임**
    - RASR에는 ASR system을 train과 decode할 수 있는 모듈을 제공함
- **RETURNN은 RASR을 extend해서 다양한 ASR system의 RNN구조를 support해주고자함**

### 1.2.3 General usage

- RETRURNN은 fully functional training software을 제공함
    - user interaction과 multi batch trainer 그리고 이후 processing을 위한 network activation을 추출할 수 있는 확률값을 제공

## 1.3 Comformer

- 해당 모델은 본 연구에서 사용된 모델이기에 간단한 리뷰를 진행함

> Gulati, Anmol, et al. "Conformer: Convolution-augmented transformer for speech recognition." *arXiv preprint arXiv:2005.08100* (2020).
> 

### 1.3.0 Overall concept

- 음성 인식은 광범위적인 이해도 중요하고, 앞뒤 단어 간 관계도 중요하기 때문에 Conformer을 사용함
- Conformer는 음성 인식에서 주로 사용되는 model임
    - **Transformer는 전체적인 의미를 잘 파악하고 CNN은 부분적 의미를 잘 파악하니까 이 둘을 같이 사용한게 Conformer**
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1ckRm5_KhqBazH9pOP9d1uuNYtzwth7vs){: width="80%" height="80%"}{: .center}
    

### 1.3.1 Comprehensive Architecture

- Convolution Module – Pointwise conv + 1D Depthwise conv (Full convolution과 동일한 효과이나 더 빠름)
- MSHA Module에는 Relative Positional encoding을 사용해서 상대적 위치 정보를 가지고 갈 수 있게
- FF Module에서는 Half step residual connection 사용
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1-7GSLqW9FeAkJXSi_cI8wAkEvJXMYKNS){: width="80%" height="80%"}{: .center}


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