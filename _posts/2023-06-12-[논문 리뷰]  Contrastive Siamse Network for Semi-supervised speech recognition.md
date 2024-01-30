---
layout: post
categories: 논문리뷰
tags: audio, ASR, semi_supervised
comments: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] Contrastive Siamese Network for Semi-Supervised Speech Recognition"
---

출처
> Khorram, Soheil, et al. "Contrastive Siamese Network for Semi-Supervised Speech Recognition." *ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE, 2022.
Copyright of figures and other materials in the paper belongs to original authors.


# 0 Abstract

- 본 연구에서는 **constrastive siamese(c-siamse) network**을 소개함으로서 **라벨링 되지 않은 Speech recognition task에서 발생하는 acoustic data의 부족을 해결**하고자 함
- c-siamse 은 **두개의 동일한 transformer encoder**으로부터의 **output 을 maching 함**으로서, speech으로부터 언어적인 **high level 정보를 추출**할 수 있음

# 1 Introduction

- 자막이 있는 대용량의 데이터셋을 수집하는 것은 시간과 비용이 많이 소요됨
- 가장 흔하게 대처할 수 있는 방법은 unlabeled data를 통해서 self/semi supervised technique을 적용하는 것
- **본 연구에서는 현존하는 self/semi supervised speech recognition technique들의 성능을 높이고자함**
- 인간의 개입이 없는 representation 학습은 3가지의 카테고리로 나뉠 수 있음
    1. input level representation
        1. degenerate solution이 없기 때문에 해당 카테고리의 학습이 조금 쉬운 편
        2. 예를 들어서 autoregressive predictive coding(APC)는 과거의 frame을 기반으로 무방향 네트워크를 사용해서 미래의 frame을 생성해 냄
        3. DeCoAR, TERA, MOCKINGJAY와 같은 경우에 masking된 input을 통해서 bidirectional network을 통해서 mask 지역을 생성해냄
        4. 이러한 방법들은 L1 reconstruction loss가 stable하고 optimize되기 쉽기 때문에 L1 reconstruction loss의 이점을 받음
        5. 하지만 input을 생성해 내기 위해서는 netowrk이 input의 detail을 학습해야하는데 이는 supervised task에서는 필수적이지 않음. 결과적으로 이러한 기술을 semi supervised framework에 적용하는 것이 적절하지 않을 수 있음
    2. intermediate level representation
        1. CPC, wav2vec, vq-wav2vec등을 포함함
        2. future input을 생성하는 대신 future intermediate representation을 predict하려고 함 
        3. intermediate represenation을 마스킹 한 후, 마스킹 된 region을 predict함
        4. 이러한 기술들은 contrastive 혹은 clustering loss를 사용하는데 뭔가 성능 향상을 위한 여지? 가 아직 남아있음
    3. output level representation
- **Speech SimCLR**
    - **해당 모델은 input과 output level prediction loss를 둘 다 사용함**
    - **augmentation module** 이 input을 두개의 **correlated view으로 transform**해줌
    - 그 후, **transformer**가 두개의 view으로부터  **output level representation을 extract**함
    - 해당 모델은 두개의 loss를 최소화 하려고 함
        1. **contrative loss ( that matches output representations)**
            1. SimCLR은 i**nput 대신 transformer의 positional embedding을 사용**하며 constrastive loss를 minimize하고자 했음
            2. 하지만 **reconstruction loss는** SimCLR으로부터 supervised task에서 consistent하는 것을 방지하였음? →**일관적인 적용이 힘들다는 말인듯**
        2. **reconstruction loss (that matches input and output representations)**
            1. 해당 loss는 **shortcut learning problem 을 해결**하고자 사용되었음
                1. shortcut learning problem이란?
                    
                    > Geirhos, Robert, et al. "Shortcut learning in deep neural networks." *Nature Machine Intelligence* 2.11 (2020): 665-673.APA
                    > 
                    
                    → 학습시키고 있는 네트워크의 학습 의도와는 다르게 loss function의 값만 줄이는 방법으로 모델이 학습되고 있는 경우를 말함
                    
- 본 연구에서는 **temporal augmentation method**를 제안하며 speech recongniton에서 모델이 **consistent**하도록 했고(일관적으로 뭔가 모델을 적용할 수 있도록 했다는 말 같음) **short cut learning problem을 효과적으로 감소**할 수 있도록 함
    
    → consistent 하다는게 section 3을 보면 확인할 수 있듯이 성능이 일정한지 아닌지를 나타내는거인듯
    

# 2 Related Work

- 본 연구는 SimCLR, BYOL, MoCo, SimSiam 방법론과 관련이 높음
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/3537651e-af2b-4979-b732-42218d31e6ab){: width="100%" height="100%"}{: .center}
    
    - 위의 architecture 들은 모두 다  2개의 branch으로부터 생성된 각각의 high level representation을 matching하고자 함
    - BYOL은 similirity loss에 기반한데 이는 **momemtum encoder을 사용**하기 때문에 collapse되지 않음
        - oneline과 target branch 으로 이루어져 있는데 traget branch는 online branch의 exponential average임
    - MoCo는 contrastive loss와 momemtum encoder을 모두 사용함
        - contrastive learning을 dictionary look up으로 생각해서 main encoder이 query representation을 추출 할 수 있도록 하고 momemtum encoder이 key의 queue를 추출할 수 잇도록 함
        - contrastive loss를 query와 그에 해당하는 key를 matching시켜주기 위해서 사용함
    - SimSiam은 같은 encoder을 각각의 branch에서 사용하는데 encoder의 output을 cosine similarity 함수를 통해서 매칭시켜줌.
        - SimSiam은 collapsing problem을 stop gradient operation과 학습가능한 projection network을 통해서 해결함
    - high level representation을 그냥 matching시켜주는 간단한 방법은 모든 output을 constant vector으로 전환하는것
    - SimCLR은 같은 data sample으로부터 생성된 두개의 다른 augmentation 사이의 agreement 를 maximaizing하면서 representation을 학습함
        - constrastive loss를 사용해서 collapsing problem 을 예방하고자 함
            - collapsing problem이란 매번 비슷한 출력값이 나오는것을 말하는 듯
        - SimCLR 연구진의 실험에서는 학습가능한 비선형 함수를 encoder의 top부분에 쌓는게 represenation의 질을 향상시키는 것에 도움을 주는 것을 확인함
- **Figure 1에 있는 모델들은 라벨링되어 있지 않은 데이터를 학습할 때에는 효과적이지만 transformer based speech recognition에서는 효율적이지 않음.**
    - main issue 는 “shortcut learning problem”임.
        - 우리가 iput sequence을 process할 때, transformer은 training loss를 최소화하려고 함
            - 이때 transformer은 input을 무시하고 positional information만을 사용해서 training loss를 줄임
                - 서로 다른 시간 t에 존재하는 데이터가 유사하지 않을 경우 negative set, 서로 같은 시간 t에 존재하는 데이터가 서로 유사할 경우 positive set 으로 분류해야하는데 이때 position embedding을 그냥 사용하게 되면 그냥 positional embedding만 사용해서 정답을 판별하게됨 → 데이터를 이해하는게 아니라 그냥 표면적인 위치 정보를 기반으로 구별해내는 문제점 발생
        - 이러한 문제점을 해결하기 위해**서 c-siam에서는 temporal augmentation을 사용함**
        - encoder에서 input을 processing하기 전에 input의 temporal 특성에 변화를 줌
- 더 나아가서 본 연구에서는 contrastive loss를 수정해서 positive 와 engative pair을 정의해주기 전에 representation을 align해주려고 함

# 3 Preliminary experiment

- **wav2vec style training이 supervised speech recogition에서 consistent한지 확인하고자 함**
- 본 실험은 2가지 step으로 이루어짐
    1. audio encoder을 wav2vec2.9을 Librilight 60k data를 사용해서 학습함
    2. inter mediate representation을 encoder으로부터 추출해서 simple classifier 을 훈련함 
        1. 각 representation이 frame level phoneme을 recognize할 수 잇는지 봄
- 실험 결과
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/1c683f75-5ca9-4ecb-b2d0-2896f5cd59a9){: width="80%" height="80%"}{: .center}
    
- Wav2Vec을 사용했을 때 위의 그림처럼 정확도가 layer 17에서 확 떨어지게 됨. 이는 wav2vec이 audio encoder의 input을 matching하려고 노력했지만 input의 phoneme을 predict하기 어려웠기 때문에 발생한 현상임.
- 이러한 현상을 해결하기 위해서는 본 연구에서 **higher level representation을 siamese network을 통해서 매칭시켜주고자** 함

# 4 Contrastive Siamese network

- **supervised part와** **unsupervised part**으로 나누어져 있으며 **해당 part들은 같은 audio encoder을 공유하고 같이 학습됨**
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/c0860197-95e2-418d-9db0-d5535af2bccb){: width="80%" height="80%"}{: .center}
    

## 4.1 Supervised network

- Supervised network은 RNN-T 기반 transformer transducer sturcture 으로 구성되어 있음
- 본 구조에서는 **input feature의 label에 대한 likelihood**는 세가지 module으로 factorize될 수 있음
    1. **audio encoder**
        1. stacked된 strided conv layer 뒤에 transformer 가 붙어있는 구조임
            1. 두개의 strided conv layer는 log mel feature을 factor 4으로 downsampling해줌
            2. transformer의 여러개의 layer을 통해서 acoustic embedding을 빼냄
    2. **label encoder**
        1. streaming transformer-XL 을 사용해서 future label을 attend하지 않도록 함
    3. **logit function→ RNNT의 forward / backward 알고리즘에서 사용됨**
        1. acoustic과 label embedding을 input으로 받아서 logit embedding을 생성함
            
            ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/01b4c70f-603b-4f79-8fe7-07a4b9e24693){: width="80%" height="80%"}{: .center}
            
            - a : acoustic
            - l : label
            - r : logit embeddings

## 4.2 Unsupervised Network

- Unspervised network는 아래와 같이 두개의 가지로 나누어질 수 있음
    1. **augmented branch**
        1. log- mel feature을 받아서 **temporal augmentation, time masking, prediction network**을 적용함
            1. prediction network은 target branch으로부터의 output을 예측하는 network
    2. **target branch**
        1. log mel feature을 audio encoder으로 passing하여 **clean한 output**을 얻음
        2. **clean 한 output을 얻는 것을 목적으로 한 branch으로 학습 파라미터를 포함하고 있지는 않음**
- **Stop gradient and prediction network**
    - 해당 구성요소는 SimSiam architecture에서 소개되었음
        - 이는 **Siamese network의 convergence property들을 향상시키고자**한 방법
    - 해당 방법을 사용했을 때 target network은 현재 학습 state으로까지의 지식을 기반으로 하여 expected output을 생성하고, augmented branch는 이러한 expected output을 matching시켜줄 수 있음
    - 본 연구에서는 5-layer transformer - xl을 사용해서 prediction network을 구성함
- **Time aligned contrastive loss**
    - **target** 과 **augmented output**을 **matching**시켜주기 위해서 본 연구진들은 **augmented branch에서 생성된 feature을 masking**하고 **masked region의 constrastive loss를 minimize**함
    - maksing은 단순하게 feature의 연속적인 region을 0으로 설정함으로서 적용함
    - contrastive loss는 softmax기반의 코사인 유사도를 기반으로 한 negative log-likelihood function으로 설정함
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/dfda0a5f-f057-4d5e-934e-37368e13c3ee){: width="80%" height="80%"}{: .center}
        
        - at는 augmented branch에서 생성된 output vector
        - qt`는 at에 매칭되는 positive target vector
        - Q는 같은 utterance의 masked region에서 랜덤하게 뽑은 negative target vector의 집합
        - sim : cosine similarity
        - 간마 : temperature parameter
    - 참고로 c-siam에서는 각기 다른 time index으로부터 positive한 pair가 나올 수 있는데 , 이는 temporal aument가 적용되었기 때문임
- **Temporal Augmentation**
    - log mel feature을 시간을 기준으로 shifting 시켜서 temporal 특성을 변형시킨 것
    - **목표는 transformer의 positional embedding으로부터 발생한 shortcut learning problem을 방지하고자 하는 것**
- Uniform Temporal augmentation
    - 일정하게 time domain의 audio signal을 랜덤하게 설정된 tempo 비율으로 compress하거나 늘리는 것
    - **WSOLA(Waveform Similarity based Overlap and Add)를 사용해서 pitch의 contour을 바꾸지 않은 채로 speech의 tempo를 선형적으로 바꿔줌**
    - 본 논문의 실험에서는 알파가 각각의 utterance으로부터 랜덤하게 뽑은 uniform distribution임. 즉 각각의 utterance에서는 tempo ratio가 다르기  때문에 우리의 audio encoder 은 modeling이 쉽지 않음.  따라서 positional counting을 피할 수 있게 됨
- Non-Uniform Temporal augmentation
    - 일정하지 않게 time domain의 audio signal을 랜덤하게 설정된 tempo 비율으로 compress하거나 늘리는 건데, speech recognition을 negatively affect하지는 않음
        - feature trajectory들에게 **time warping function을 통해서 이를 적용함.**
            - x(t)는 time t에서의 speech feature을 의미하고, 이를 x(w(t))으로 변환함.
            - w(t)는 time warping function
            - 이를 target branch의 output에도 적용함
        - warping function은 log mel trajectory들을 보존하기 위해서 아래 세가지 제약을 지켜야함
            1. **Monotonicity**
                1. w(t)는 monotonically 증가되어야함. 그게 아니면 input sample들의 순서가 무시될 것
            2. **Smoothness**
                1. warping function의 급작스러운 변화는 feature trajectory들의 overall shape을 망가뜨릴 것
            3. **Boundary conditions**
                1. warping function은 반드시 시점 0에서부터 시작해야하고(w(0)=0) 마지막 프레임에서 종료되어야 함
                2. w(T-1)=T-1 이 지켜져야함. T는 input frame의 개수
                3. boundary condition을 통해서 input을 모두 포함할 수 있도록 함
        - time warping function
            
            ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/429822f9-613b-4072-a1ae-5ff6c13cf4af){: width="80%" height="80%"}{: .center}
            
            - R은 warping function의 순서이고, ar는 r번째 sin 구성요소의 amplitude(진폭)을 나타냄
                - 이러한 파라미터들은 smoothness랑 mononicity를 조절할 수 있음
            - time warping function을 생성한 이후에는 input feature에 이를 적용해야함. 본 연구에서는 linear interpolation 기술을 통해서 x(w(t))를 계산함
                
                ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/1598e52c-9416-478f-8eda-83c562008f3e){: width="80%" height="80%"}{: .center}
                
                - ┌w┐와 └w┘는 w의 ceil 값
    

# 5 Experiment

## 5.1 Experiment results

- 실험 결과
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/7b2cd288-231f-4855-a1c5-f53c33082ccd){: width="60%" height="60%"}{: .center}
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/d2dcf5f6-d18c-4c07-a83c-348977ee1dda){: width="60%" height="60%"}{: .center}
    
    - 실험 결과 model size는 proposed model이 작은데 거의 동일한 WER을 도출해 냄

# 6 Conclusion

- 본 연구에서는 c-siam network을 제안하였는데, 이는 semi supervised speech recognition system의 새로운 훈련 방법임
- c-siam은 supervised RNN-T 모델과 unsupervised siamese network을 동시에 훈련함
    - siamese network은 타겟과 augmented branch를 포함하고 있음
    - clean 과 augmented representation을 target과 augmented branch으로부터 추출함
    - 이후 augmented representation을 clean representation과 correlated 되도록 contrastive loss를 통해서 augmented representation을 수정함
    


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