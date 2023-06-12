---
layout: post
categories: 논문리뷰
tags: audio, voice_conversion, zero_shot
comments: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] Robust disentangled variational speech representation learning for zero-shot voice conversion"
---

> Lian, Jiachen, Chunlei Zhang, and Dong Yu. "Robust disentangled variational speech representation learning for zero-shot voice conversion." *ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE, 2022.
Copyright of figures and other materials in the paper belongs to original authors.

# 정보

- Disentangled : speaker의 정보와 content 정보가 얽혀 있는 speeah data를 구분해준다
- representation learning : high dimension data를 low dimension data의 공간으로 mapping하고자 함
- zero shot voice conversion : 새로운 speech data에도 flexible하게 speaker와 content 정보를 구분해주고, 새로운 sepach data로도 만들 수 있음

# 0 Abstract

- 좋은 음성 변환의 질은 좀 더 좋은 alignment module이나 expressive mapping function을 통해서 얻어질 수 있음
- 본 연구에서는 zero shot Voice Conversion(VC)를 제안함
    - self supervised disentangled speech representation learning 방법
- 특히 본 연구에서는 global speaker representation과 VAE의 time varying content represetation의 information flow의 균형을 맞추어주며 disentanglement를 achieve함
- zero shot voice conversion은 VAE decoder에 arbitrary speaker embedding과 content embedding을 입력값으로 넣음으로서 perform됨
- 이외에도 on the fly data 증강 training 전략을 사용해서 학습된 representation의 noise를 invariant하게 하도록했음

# 1 Introduction

- VC는 source speaker의 non linguistic information을 target speaker으로 자동적으로 변환해줌
    - linguistic content는 변하지 않음.
    - non linguistic information과 같은 경우에는timbre, emotion, accent, rhythm, .. 을 포함함
- 현재의 VC system을 두가지 방법론을 기반으로 카테고리화 할 수 있음
    1. conversion model을 통해서 source acoustic feature을 target acoustic feature으로 변환
        - 관례적인 VC의 접근에서는 acoustic feature은 source와 target 발화로부터 추출된 첫번재 acoustic feature이며, 해당 acoustic feature들은 alignment module을 통해서 frame wise하게 align되었음
        - 하지만 연구들에 따르면 alignment step은 seq2seq model을 통해서 source와 target acoustic 을 직접적으로 mapping함으로서 생략될 수 있으며 이런 경우 좀 더 좋은 VC 성능을 보인다고 함
        - VC를 nonparallel training data 에 직접적으로 mapping하는 것은 GAN based many-to-many VC system의 성능을 향상시켰음
            - 해당 방법이 광범위하게 연구되었지만, 해당 방법은 source target 의 speaker VC pair이 이미 알려진 상태를 가정하는데 이건 real world 에 적용하는데 큰 한계점을 지니게 함
    2.  VC를 emplicity learned speaking style과 content representation들에 기반하여 구성하는 방안으로, 1번의 한계점을 해소하고자하였음
        - Phonetic PosteriorGrams(PPGs)가 speaker의 independent한 content representation으로 광범위하게 사용되었음
        - pre trained speaker verification model으로부터 추출된 speaker embeding은 timbre information을 포함한다고 종종 가정되었음
        - encoder은 speaking style과 content information을 laent embedding으로 compress하고, decoder은 speaking style embedding과 content embedding을 combine해서 voice sample을 생성해 냄
        - 좀 더 좋은 VC성능을 얻기 위해서는 해당 모델들은 발화에 대한 positive pair을 얻어야함
            - 같은 speaker으로부터 녹음된 두개의 발화
- 본 연구에서는 speaker identity conversion 문제에 focus함
    - VAE를 backbone framework으로 설정하고 확장하여 disentangled content representation과 speaking style representation을 학습하도록 함
        - VAE의 학습 중에 content과 style information flow가 balancing됨
    - vanilla VAE loss를 확장해서 speaker과 content component들의 disentaglement를 강화시켜줌
    - 추가적으로 learned representation을 background 노이즈나 음악 그리고 interfering하는 speaker으로부터 robust하게 함
    - on-the-fly data 증강 기법이 VAE training의 inductive bias을 위해서 사용되었는데 이러한 training 기법을 사용하면 denoising disentangled sequential VAE(D-DSVAE)을 구축할 수 있음
        - D-DSVAE는 노이즈가 추가된 speech가 input으로 들어오더라도 이 또한 잘 처리할 수 있음

# 2 Proposed Methods

- Notations (논문에서 사용될 기호 및 backbone 모델 소개)
    - speech segment 변수 X
        
        ![Untitled](http://drive.google.com/uc?export=view&id=1_cpHAi0jk24pWFE0prZQHqTYpZKrpGA3){: width="30%" height="30%"}{: .center}
        
    - speaker **style** latent representation
        
        ![Untitled](http://drive.google.com/uc?export=view&id=1POytM655px6yh-z8CMOaln3b1uGzh-nv){: width="10%" height="10%"}{: .center}
        
    - speaker **content** latent representation
        
        ![Untitled](http://drive.google.com/uc?export=view&id=1qaA6Y1oNQ5XJzlX303EUFPdHE7fXHNAx){: width="50%" height="50%"}{: .center}
        
    - backbone - DSVAE
        
        ![Untitled](http://drive.google.com/uc?export=view&id=1AyjH6uEZd2b6HbqxOqghNkcWnFjJG3Q3){: width="80%" height="80%"}{: .center}
        
        - X를 input으로 받으면, encoder Eshare(ES) 을 통해서 W를 생성해 냄.
            - speaker encoder Econtent(EC)는 W를 input으로 받고 posterior 분포를 모델링함
                
                ![Untitled](http://drive.google.com/uc?export=view&id=1egpQ9-PNMhUnMDFrMLXtq0K4QpunFbZr){: width="60%" height="60%"}{: .center}
                
                - Zs랑 Zc는 q(Zs|X)와 q(Zc|x)를 샘플링 해서 얻어짐
                - Zs는 오직 speaker information을 encoding하고, Zc는 오직 content information을 인코딩하기를 기대함
            - generation stage에서는Zs와 Zc가 concat되어서 shared decoder D에 입력됨.
            - D share에서는 spectrogram X^를 생성해 냄
            - vocoder은 X^를 waveform으로 변환해줌
        - source X1을 target X2로 변환할 경우, converted speech는 X^12 = D(Zs2, Zc1)이 됨

## 2.1 disentanglement Aware Probabilistic Graphical Models

- 두개의 component 간의 disentanglement를 achieve하기 위해서 자주 사용되는 방법은, 서로가 확률적으로 독립이라고 가정하는 것임.
- 본 연구에서는 Zs (speaker **style** latent representation)와Zc 의 prior distribution과 postor distribution을 independence가정에 따라서 factorize함
    - Z = [Zs, Zc] 이며 Z는 joint latent representation이라고 denote함

### Prior

- joint prior 분포는 다음과 같이 factorised됨
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1P8hrP-h8JTxoVDUc4x70T9fCPmvkJCZV){: width="60%" height="60%"}{: .center}
    
    - p(Zs)는 표준 정규 분포 N(0, Ids)를 따르고 p세타(Zc)는 autoregressive LSTM을 통해서 얻어짐

### Posterior

- joint posterior 분포는 다음과 같이 factorised됨
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1dUwXY8TISYKrqWrt93hBAR_drRbYoDc2){: width="60%" height="60%"}{: .center}
    
    - 각각의 q세타(Zs|W)와 q세타(Zc|W)는 두개의 독립적인 LSTM을 통해서 모델링됨
    - 여기서 W는 Eshare의 출력값임

## 2.2 Loss Objectives

- 먼저 아래 식은 vanilla VAE loss에 관한 식임
    
    ![Untitled](http://drive.google.com/uc?export=view&id=17pWip3yo12vBWB03vy78tSo63bufW0xM){: width="80%" height="80%"}{: .center}
    
    - kl은 두 분포 간의 KL divergence을 의미
    - 알파와 베타는 balancing factor을 의미
    - loss obj는 Zs와 Zc를 직접적으로 구분하도록 enforce하지는 않지만, 추후 설명될 세 단계로 나뉘어서 Zs와 Zc를 구분하게 됨
    - 알파kl(p(Zs)와 관련된 식의 2번재 항은 speaker Zs의 prior p(Zs)와 실제 데이터 X에 대하여 학습된 q세타 간의 KL Divergence임
    - 베타kl(p(Zc)와 관련된 식의 3번재 항은 content Zc의 prior p(Zc)와 실제 데이터 X에 대하여 학습된 q세타 간의 KL Divergence임
    - 

### 2.2.1 Variational Mutual information and KL vanishing in VAE

- VAE의 original form에서는 mutual information MI(X, Z)는 아래 식과같이 표현되었음
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1j80myI9OTqzRhnnPKQqBuhKKX8WgZdSk){: width="30%" height="30%"}{: .center}
    
- 그리고 해당 kl의 변형은 아래와 같이 나타내었음
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1tUjSFjrfTOrmlnLayfbG4PJyxks5LMNs){: width="80%" height="80%"}{: .center}
    
    - 위의 식은 variational mutual information이라고 불림
- VAE의 objective는 아래와 같이 설정됨
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1szkm5leMtzXFpAsJErhDr3Z2D-UWnPGV){: width="80%" height="80%"}{: .center}
    
    - 해당 식의 앞부분은 reconstruction loss이고 뒷부분은 Variational mutual information을 나타냄
    - reconstruction loss가 낮다는 것은 variational mutual information이 높다는 것이기 때문에 해당 파트를 동시에 최소화하는 것이 일반적으로는 달성될 수 없음
        - 해당 식에서 reconstruction loss를 줄이기 위해서는 -log안 속의 P세타(X|Z)가 1에 가까워 져야 함. 이는 X와 Z이 서로 연관이 많아졌을 때 1에 가까워질 수 있는데, 이 말은 Mutual information이 당연히 커지게 됨
    - 그런데 decoder이 강력한 self supervised generation 능력을 가지고 있다는 가정 하에는 위의 목적이 달성될수야 있지만 본 연구에서는 그정도로 강력한 디코더를 가지고 있지 않는 상황을 고려함

- 따라서 VAE는 reconstruction loss와 variational mutual information loss 를 적절하게 잘 balancing하고자 하며 아래 식과 같이 variational mutual information이 lower bounded를 따르도록 함
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1LaAw8dAmNquCZpDBOYeLEvY6P-p0Bk-i){: width="50%" height="50%"}{: .center}
    
    - A(X,Z)는 bounded variational mutual info를 의미함
    - 해당하는 식 덕분에 VAE training 시에 KL vanishing이 발생하지 않게 되는데 이는 disentanglement의 기반이 되게 한다
        - 

### 2.2.2 Information Flow Between Multiple Latent Variables in VAE

- 본 연구에서는 Zs와 Zc간의 information flow가 balancing되어 있으면 disentanglement가 성립할 수 있다고 주장함
- 아래 식은 본 연구에서 주장한 loss함수임
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1K-JOUeSLqarkBZrcYXkCh3OSyiDSU7jJ){: width="80%" height="80%"}{: .center}
    
    - 2-2-1에 따라서 Zs와 Zcc를 통해서 각각 인코딩된 정보를 summation하는 것은 input speech를 reconstruct하는데에 충분함
        
        ![Untitled](http://drive.google.com/uc?export=view&id=159YE0RfAc2gRC9Q9_cKXSMf8tlH8L14Y){: width="80%" height="80%"}{: .center}
        
        - 식을 조금 풀어서 보면, inf 기호가 하한값에 관한 기호이니까 speaker와 contents에 관한 mutual information을 더한 하한값은 적어도 A(X,Z) 즉 bounded variational mutual information보다는 크거나 같아야한다는 것
- 수렴을 가정한다면, 베타 나누기 알파가 엄청나게 클 경우에는 두개의 variational mutual information term이 M^I(X,Zc)에 의해서 결정될 것임. 이는 KL(q(ZC|X)||p(XC))를 vanishing하게 하여 latent variable으로 encoding 된 정보가 Zs에 의해 결정되게 할 것.
    - 조금 풀어서 생각을 해보면, X와 Zc간의 mutual information이 영향을 크다는 것은 loss를 줄이면서 학습할 때 해당 값을 줄이기 위해서 학습이 됨. 따라서 X와 Zc의 mutual info가 작아지도록 학습하면 X와Zc의 상관관계가 작아지게 되고, 이로서 X와 Zs간의 mutual info가 남게 되니까 Zs에 의해서 encoding이 됨.
- 연구진들은 실험적으로 variational mutual information loss가 reconstruction loss에 독립적이라는 것을 확인했음
    - 이러한 경우에는 Zs와 Xc사이에 양면적인 정보 흐름이 있다고 볼 수 있고, 적절한 베타와 알파를선택하면 인코딩 정보간의 overlapping된 부분이 없을 수 있음

### 2.2.3 Time invariant and Time variant disentangled

- 연구진들은  speech feature sequence의 time 차원에 average pooling을 적용하면 disentangle을 achieve 할 수 있다고 주장함
- 2.2.2에 따라서 M^I(Zs, Zc)=0이더라도 어떤 latent variable이 speaker information이나 content information을 encoding한다는 것은 모름
- 본 연구에서는 speaker feature들을 average pooling해서 speaker information을 유지하고자 했음

# 3 Experiments

## 3.4 Results and discussions

- 아래 figure에서 확인할 수 있듯, 베타나누기 알파가 1일때는 발화의 절반만 target speaker으로 변환된 것을 확인할 수 있음. 즉 Zs보다는 Zc를 통한 변화가 보임
    
    ![Untitled](http://drive.google.com/uc?export=view&id=12P5Roqnnnz5tqb2bDcQrLXYn412ydjv5){: width="80%" height="80%"}{: .center}
    
    - 값이 100일때는 반대로 Zc는 아무런 역할을 하지 않고 Zs으로 인해서 결과물이 창출된 것을학인할 수 있음



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