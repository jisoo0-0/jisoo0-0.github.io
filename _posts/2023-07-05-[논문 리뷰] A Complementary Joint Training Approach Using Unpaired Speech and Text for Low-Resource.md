---
layout: post
categories: 논문리뷰
tags: audio, asr
comments: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] A Complementary Joint Training Approach Using Unpaired Speech and Text for Low-Resource Automatic Speech Recognition"
---

출처
>   Du, Ye-Qian, et al. "A Complementary Joint Training Approach Using Unpaired Speech and Text for Low-Resource Automatic Speech Recognition." *arXiv preprint arXiv:2204.02023* (2022).
Copyright of figures and other materials in the paper belongs to original authors.

- 용어 정리
    - **Pseudo labeled data**란?
        - 모델을 통하여 라벨이 없는 데이터에 라벨을 붙인 데이터
    - **synthesized audio text pair**이란?
        - 아마도 모델을 통해서 pair으로 짝지어진 audio text pair
- first round, secound round
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/05166275-df91-43f0-ae70-0a287ba424f9){: width="40%" height="40%"}{: .center}
    
    - first round 에서는 model을 두개의 paired generated dataset을 통해서 학습시킴
    - second round 에서는 generated된 pseudo label들을 masking하고, gradient restriction을 synthsized speech data 에 적용하고, model을 re training시킴

# 0 Abstract

- 자원이 풍부하지 않은 ASR(Automatic Speech Recognition) 에서 unpaired data는 beneficial한 결과를 보여주고 있음
- 본 연구에서는 unpaired data를 통해서 일반적인 sequence to sequence model을 학습하려고 함.
- Unpaired speech와 text data는 data pair을 형성하기 위해서 사용됨
- speech-PseudoLabel pair과 Synthesized Audio-text pair의 복잡성으로 인해서 본 연구에서는 **complementary Joint Training(CJT) 방법을 사용해서 model을 두개의 데이터 pair들을 번갈아가면서 학습시킴**
- **real data으로의 적용을 위해서 합성된 audio에 gradient restriction과 pseudo label에 대한 label masking이 적용됨**

# 1 Introduction

- end to end 구조는 ASR의 주요한 paradigm으로 남아있음.
- single network 구조는 training process와 joint optimization가 기존 관습적인 모델들보다는 훨씬 간단하지만 성능은 impressive함
    - 그럼에도 불구하고 labeled data가 엄청 많이 필요한 것은 사실임
    - labeled data는 생성하기 너무 까다롭고 비용도 많이 들고 시간도 많이 드는게 명백한 단점
- 본 연구에서는 자원이 부족한 ASR task에서 어떻게 unpaired data를 사용할 것인지에 대해서 focus 하도록 하겠음
    - 여기서 unpaired data는 audio와 text 각각의 unlabeled된 데이터를 말함
- unpaired data의 사용에 대해서는 엄청나게 많은 researh가 진행되었음
    - speech only data와 같은 경우 흔한 approach가 unsupervised training 을 feature extractor으로 사용하는 것 혹은 pseuso label을 통해서 self traiing을 하는 방식임
    - text only data 와 같은 경우 외부 language model(LM)을 joint decoding으로 사용해서 text를 학습함
    - both unpaired data를 사용한 연구는 pre-trained acoustic model과 LM을 같이 사용하거나 representation learning을 공유하는 것 등이 있음.
        - 이들은 multi task training을 하는 hybrid model에 의존하고 라벨링  데이터의 양이 제한적이기 때문에 less effective하기도 함
        - 현재 SOTA 모델은 speech 와 text data를 사용해서 pre training하고, LM을 joint decoding을 위해서 training한 모델임
        - 하지만 이러한 방법들은 LM의 능력을 모두 사용하기 위해서 large beam search space를 요하기 때문에 계산량이 너무 많아지게 됨
- unpaired speech, text data를 통해서 일반적인 seq2seq모델을 훈련하기 위해서 partial pre training이 적용될 수 있음.
    - **다만 pre trianing과 fine tuining의 불일치가 모델의 성능에 제약을 줄 수 있음**
- 이러한 문제를 해결하기 위해서 본 연구에서는 ASR 모델을 sample pair을 이용해서 training함
    - sample pair은 pseudo labeling과 TTs synthesis를 통해서 구성됨
    - low resource 시나리오에서 생성된 데이터는 real data와 확연히 다른 경우가 많기 때문에 **speech PseudoLable pair**이나 **Synthesized Audio Text pair**들을 각각 홀로 사용하는 것은 model training에 심각한 제약을 줄 수 있음.
    - 두개의 data pair가 input acoustic feature과 output linguistic feature 측면에서 보완적(complementary)하기 때문에, 두 데이터 pair을 통해서 model을 training하고자 하고 이러한 방법을 complementary joint training(CJT)라고 부르기로 함.
        
        **→ speech psesudo label pair혹은 synthesized audio text pair을 각각 단독으로 사용하는 것은 문제가 되지만 해당 데이터들을 같이 사용하여 학습하는 것은 상호 보완적이기 때문에 괜찮음**
        
    - basic CJT에서 2가지 전략을 추가해서 CJT ++으로 부름
        - pseudo-label중에서 low first-round confidence을 가지는 token을 마스킹함
        - **synthesized audio 중에서 lower layer의 gradient back propagation을 부분적으로 block해서 실제 audio에 잘 적용될 있도록 함**

# 2 Complementary Joint Training

## 2.1 Basic Complementary Joint Training

- basic CJT 에서는 다양한 unpaired speech와 text data가 training에서 사용되었음.
    - paired data 중 일부를 사용해서 data preparation을 진행하였음
- 수식 정리
    - Dp 는 paired speech-text data
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/2c1e4599-4001-48cb-babc-5530a196ad93){: width="30%" height="30%"}{: .center}
        
    - 아래 식들은 각각 unpaired speech data, unpaired text data
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/ed6d11be-9d53-4246-b5f6-926eb47edc16){: width="20%" height="20%"}{: .center}
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/135a582c-d996-403c-a3ae-fa721cc1ebc9){: width="20%" height="20%"}{: .center}
        
    - audio x*을 TTS를 통해서 찾아냄, y는 text data sample x*은 그로부터 생성된 sample speech data
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/bff3dd3a-8460-4397-a717-eb1bbeb0f980){: width="20%" height="20%"}{: .center}
        
    - SynA-text pair은 아래와 같이 표기함
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/e8780e20-abe6-48ec-ab81-735410bcfb03){: width="30%" height="30%"}{: .center}
        
    - CJT model은 Dsu와 Dtu을 통해서 alternatively 하게 updated됨
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/ca9a1661-cbec-4516-824a-2d39e1e21140){: width="20%" height="20%"}{: .center}
        
        - Ls : speech-PseL pair에 대한 loss
        - 람다Lt : SynA-text pari에 대한 loss
    - Ls, Lt구하는 식
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/9c53fee0-e425-4ea0-b3b1-816fc249aad7){: width="40%" height="40%"}{: .center}
        

## 2.2 Analysis and two enhancement strategies

- speech-PseL과 SynA-text pair들의 온전한 특성을 밝혀내기 위해서 잘 맞춘 그리고 잘못 맞춘 token을 아래 figure 2 와 같이 확인해 봄
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/216265d2-1124-4ce3-b696-bd3f9dd75559){: width="60%" height="60%"}{: .center}
    
    - fig2의 윗부분은 speech-PseLdata 에만 training한 것, 아랫부분은 speech-PseL과 SynAtext data에 jointly training한 것
    - 결과적으로 보면 윗부분에서는 correct / incorrect token이 비슷한 분포를 보이지만 아래 쪽은 incorrect token을 생성해 내는 비율이 완전히 내려감
- 더 나아가서 PWCCA를 통해서 encoder layer의 representation의 similarity를 측정하고자 했음
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/fbab7703-39e9-42d4-8c18-699c4ccaba5a){: width="60%" height="60%"}{: .center}
    
    M: 100h paired data 
    
    M-real: 100h speech-PseL data 
    
    **M-syn: 860h synA-text data** 
    
    M-joint: 100h speech-PseL data + 860h synA-text data
    
    ⇒ PWCCA는 satndard model을 960h paired data에 훈련시킨것을 기반으로 계산됨
    
    - TTS synthesized audio를 학습하는 것은 deviation을 명백하게 발생시킨다는 것을 확인할 수 있었음
        - M-syn의 곡선이 layer 가 깊어질수록 simliarity 가 낮아지는 것을 볼 수 있음
    - synthesized audio 와 real audio 를 모두 사용해서 training 한 모델은 낮은 layer에서 ground truth 의 feature들과 비슷한 feature을 보임
        - M-real이 ground truth data이니까 위의 figure3에서 비슷한 양상을 보이는 것을 확인 가능
    - higher layer에서는 linguistic modeling의 능력이 강해지는 것을 확인할 수 있었음.

### 2.2.1 Label masking for pseudo-labels

- pseudo label들에 training하는 것은 model을 overfitting하게 할 수 있음
    
    → 이게 pseudo label이 어쨌든 모델이 생성한 데이터니까.. overfitting될 수 있다는 것
    
- additional text는 model이 incorrect token들에 대해서 보다 더 discriminative하게 대응할 수 있도록 도와줌 (fig 2의 결과였음)
- 따라서 본 연구에서는 prediction 확률값을 incorrect target token을 구별하는데에 레퍼런스로 사용하게 되고, label masking 전략을 제안함
    - 라벨 마스킹 전략은 first round prediction probability값이 낮은 토큰을 마스킹 하는 것
    - 이렇게 하면 incorrect label에 overfitting하는 것을 막아주고, context modeling을 향상시킬 수 있음
        
        → historical info가 없으니까 그렇게 된다고 주장
        
- pseudo label squence에 대해서 T가 target sequence의 길이임
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/ee81d242-149a-4f04-a766-72c34a60899f){: width="20%" height="20%"}{: .center}
    
    - 본 연구에서는 binary mask sequence을 생성해서 몇몇개의 token들을 들을 <PAD>으로 교체해줌
- speech-PseL pairs 의 두번째 training 의 loss
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/9aa666ce-019a-4606-8759-0af00eea71ca){: width="50%" height="50%"}{: .center}
    
    - masked target sequence 은 y~*으로 표기하고 mask index들의 집합을 M임
    - 첫번째 round prediction에서 예측된 확률값 p 를 기반으로 세개의 masking 기법이 고려됨
        1. confidence 기반 masking(conf)
            1. token이 k*(1-p)의 확률로 masking됨 
            2. k는 mulitplier임
        2. threshold 기반 masking(thres)
            1. 만약 p가 일정 수치를 넘지 못하면 masking됨
        3. random 기반 masking(rand)

### 2.2.2 Gradient restriction for synthesized audio

- TTS synthesized audio가 real audio보다 더 작은 변동성을 가지고 있기 때문에, synthesized data의 사용은 실제 speech에 ASR을 적용했을 때 더 낮은 성능을 보이게 됨
- 하지만 real audio와의 joint training을 통해서 model이 더 넓은 acoustic feature extraction을 해낼 수 있게 됨(Fig 3의 결과)
- synthesized audio가 작은 deviation을 도래하지만 본 연구에서는 gradient restriction training 기법을 사용해서 feature mismatch을 줄이려고 했음
    - 이를 위해서 얕은 레이어 층에서 gradient propagation을 랜덤하게 blocking함
    - real audio의 acoustic feature을 추출해 낼 때 좀 더 잘 fitting가능
    - 본 연구에서는 처음 4개의 layer들이 shallow layer(얕은 레이어층)으로 구분됨
    
    

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