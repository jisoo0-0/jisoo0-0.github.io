---
layout: post
categories: 논문리뷰
tags: ASR, GAN, unsupervised-learning
comments: true
math: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] Enhancing Unsupervised Speech Recognition with Diffusion GANS"
---


**논문 및 사진 출처**
>Wu, Xianchao. "Enhancing Unsupervised Speech Recognition with Diffusion GANS." ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023.

# Abstract

- 본 연구에서는 vanilla adversarial training method for unsupervised ASR 을 diffusion GAN을 통해서 enhancing하고자 함.
- 본 연구가 제안한 모델은
    1. generator의 output과 unlabeled reference text에 instance noise 를 추가함
    2. diffusion 모델에  timestep-dependent discriminators을 요청함. 
    3. generator을 업데이트하기 위한 backpropagate진행

# 1 Introduction and background

## 1.1 Unsupervised ASR

- 본 연구에서는 unsupervised version의 wav2vec을(wav2vec-U) 사용함.
    - wav2vec-U에서는 아래와 같이 2개의 line이 존재함
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/cc730fef-c967-4c7e-b4c9-ff151caa1094){: width="90%" height="90%"}{: .center}
        
        - first line
            - unlabeled speech audio를 받아서 wav2vec에 audio를 speech representation으로 encoding함. 그리고 k-means 를 사용해서 각 audio segmentation에 대한 각 시간의 representation을 클러스터링하고, PCA를 통해서 representation을 compress함
            - 이후 mean pooling을 통해서 비슷한 cluster id들은 합병됨
            - 추가적으로 인접한 segement들에 대해서 mean pooling을 진행해서 미래의 음소 시퀀스들의 길이를 어느정도 발랜싱해줌
        - second line
            - unlabeled text를 수도로 음소화 하고, one-hot representation으로 reshape함
    - 두개의 line에서 생성된 각각의 representation들은 discriminator으로 전달되어 adversarial training됨
- wav2vec-U의 objective setupt에서는 G와 C는 각각 다른 (S,P^r) pair으로부터 최적화됨.
- Generator 의 차례에서는 loss가 아래와 같이 minimized됨
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/4aa81799-d297-4a65-bf4c-3dd68260e39f){: width="30%" height="30%"}{: .center}
    
    S: segment sequence
    
    P^r : “pseudo” reference phoneme sequence
    
    - L_pd는 phoneme diversity loss임
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/bff3e18b-3afc-4cc8-89be-2804160fe96a){: width="30%" height="30%"}{: .center}
        
        - Batch B 레벨에서 음소 entry가 잘 사용되지 않을때 G를 penalizing하도록 설계됨
        - H_g(g(s))는 averaged softmax distribution의 entropy임
    - L_sp는 segment smoothnes penalty loss임
        - genrator가 인접한 sement에 대해서 더 비슷한 output을 출력할 수 있도록 함
            
            ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/30c56c05-f3cf-4b7a-a7b7-ffe6328e224a){: width="30%" height="30%"}{: .center}
            
- Discriminator의 차례에는 아래와 같이 loss가 minimized됨
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/1b0d617d-4d6e-447e-9c95-cf4ac0b056cd){: width="40%" height="40%"}{: .center}
    
    - L_gp는 gradient penaly로, input의 관점에서 discriminator의 gradient norm에 대한 패널티를 부여함
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/954dacf9-3142-4b69-8cfb-6cf586f7a41b){: width="30%" height="30%"}{: .center}
        
    - Input sample P~P는 stocastic linear combination of the activations of pairs of real and fake samples임.
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/07d4e632-6239-44a1-8c26-c26569746c30){: width="20%" height="20%"}{: .center}
        

## 1.2 Diffusion GANs

- Wav2vec -U framework에서 pseudo textual reference는 text set으로부터 샘플링됨.
    - 근데 이 text set은 음성 dataset과 완전히 일치하지는 않을 수 있음. →당연한것..
- vanishing gradient는 data와 generator distribution이 너무 다를때 발생함.
- 여기서 우리는 training instability 및 mode collapse 문제에 대해서 해결해보고자 함.
- 본 연구에서는 아래와 같은 문제를 대답해보고자함.
    - will diffusion process가 training stability와 mode diversity를 개선시킬 수 있나?
        - 이걸 대답하기 위해서 현존하는 diffusion -GAN framework을 사용함.
        - diffusion-GAN framework은 아래와 같은 구성요소가 포함되어 있음
            - adaptive diffusion process
            - diffusion timestep-dependent discriminator / generator
- 매 timestep t마다 t-dependent dicsriminator C(y,t)가 diffused real data와 diffused generated data를 구분해내는 것을 학습함.
    - G는 각각의 C(y,t)의 feedback을 통해서 학습함.
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/3dde2dee-caa1-4dfd-89d8-89e4c219066f){: width="50%" height="50%"}{: .center}
        
        - x~p(x)와 p(x)는 true data distribution
        - t~p_pie 와 p_pie는 각 diffusion timestep t에 대해서 각기다른 weights pie_t를 할당시켜줌.
        - y~q(y|x, t)는 y가 조건부 q(y|x,t)에서 sampling된다는 것
        - q(y|x,t)는 변형된 샘플 y의 조건부 확률 분포를 나타냄
            - data x, timestep t에 관함

# 2 Diffusion-GAN enhanced WAV2VEC-U

- 본 연구의 주 목적은 wav2vec-U에 사용된 GAN의 objective를 difussion-GAN의 objective으로 변경하는것(식 1,2를 식3으로 변경)
- BERT의 역할
    - phonemes의 contextual 한 distribution을 학습하고자함
    - 이러한 distribution을 통해서 pseudo phoneme sequence의 controllable sampling을 진행하고자 함.
- 주로 G(s)와 P^r이 상대적으로 high diemnsion vector으로 표현되기 때문에, unsymmetrical U-net을 사용해서 down 및 upsampling을 진행함.
- **Generator는 wav2vec-U를 상속받고 non-causal conv layer1개를 포함함. original discriminator을 통해서 generator의 prediction을 구분해냄. 주요 updated 된 부분은 additional T+1 timestep dependent discriminator을 사용해서 diffused real data y으로부터 diffused generated data를 구분해 내는것임.**
    - The generator learns from all these T + 2 discriminators’ feedback by back-propagating through the forward diffusion process.
- Diffusion-GAN의 theorem
    1. adding noise to the real and generated samples 가 학습에 도움이 됨
    2. minimizing the JS divergence between the joint distributions of the noisy samples and intensities can lead to the same optimal generator as minimizing the JS divergence between the original distributions of the real and generated samples
        
        → 잡음이 있는 데이터에 대해서 학습하면 original distribution을 학습하는 것과 같은 효과를 낼 수 있음
        
    - 위 두개의 theroem을 통해서 diffusion-GAN은 noise을 통해서 GAN의 학습을 강화하는 것을 알 수 있었음
- 오디오 시퀀스의 길이와 semantic하게 같은 길이를 가지는 문장을 생성하기 위해서 phoneme-based pretrained LM을 사용함.
    - we first **obtain the number of segments** after pre-processing the audio using wav2vec2.0 representing + k-means clustering + segment merging, and **then use this length to guide the length of sampled phoneme sequence from M**.
    - This P r sampling method has **several benefits during training**: controllable pseudo reference text length, **a better coverage of the target phoneme distribution**, and a dense instead of the original one-hot representation o**f P r with richer context**s.

# Ablation

Ablation에서 나타난 각 모듈의 역할

- Bert
    - length guiding
    - phonemes contextual distribution 학습
- Length guiding
    - training stability
- T+1 diffusion timestep-dependent discriminators
    - Distinguishing between generated and real data at each diffusion timestep.
    - Trained to separate generated and real data at each timestep of the diffusion
    process

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