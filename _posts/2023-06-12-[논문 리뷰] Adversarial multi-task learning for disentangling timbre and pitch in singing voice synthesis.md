---
layout: post
categories: 논문리뷰
tags: audio, voice_synthesis
comments: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] Adversarial Multi-Task Learning for Disentangling Timbre and Pitch in Singing Voice Synthesis"
---

출처

> Kim, Tae-Woo, Min-Su Kang, and Gyeong-Hoon Lee. "Adversarial Multi-Task Learning for Disentangling Timbre and Pitch in Singing Voice Synthesis." *arXiv preprint arXiv:2206.11558* (2022).


Copyright of figures and other materials in the paper belongs to original authors.


# 0 Abstract

- deep learning을 기반으로 한 generative singing voices들이 많이 연구되고 있음
    - 노래하는 목소리를 생성하는 task을 수행하는 한 가지 방법은 **명백한 speech parameter을 포함하고 있는 parametric vocoder feature을 예측**하는 것임
        - 해당 접근 방식은 각각의 feature가 명백하게 구분될 수 있다는 이점을 가지고 있음
        - 해당 접근 방식의 한계점은 **voice의 quality가 좋지 않다**는 것
    - 또다른 접근 방법으로는 **neural vocoder을 위한 mel spectrogram을 예측**하는 것임
        - 해당 접근 방식의 한계점은 **timbre과 pitch의 정보가 entangled되어 있**기 때문에 modeling이 어렵다는 것
- 본 연구에서는 **parametric vocoder feature을 auxiliary feature으로 사용**해서 제안된 모델이 **효과적으로 mel spectogram의 timbre와 pitch 구성요소를 잘 구분**해 낼 수 있도록 했음. 더 나아가서 **GAN framework**이 **multi singer model**에서의 **singing voice의 퀄리티를 증진**시키기 위해서 적용되었음
    - timbre : 음색
    - pitch : 음의 높낮이

# 1 Introduction

- **SVS(Singing Voice Synthesis)**는 **singing voice를 musical score과 lyric에 따라서 합성**하는 생성모델임
- musical score이 note pitch와 duration information을 제공하지만, 일반적으로 발화하는 speech voice 보다 **더 긴 vowel duration과 wider pitch range을 가지고 있기 때문에** singing voice를 모델링하는게 어려움.
    - 더 나아가서 singer의 vocalization과 expressin에 따라서 pitch information이 바뀔 수 있고 timbre 가 있다는 점이 까다로움
- **singer 의** timbre **특성은 voice source이랑 vocal tract에 따라서 달라**지며, natural singing voice를 모델링하기 위해서는 **SVS가 timbre와 pitch간의 interdependence를 고려해서 모델링**되어야함
    - 최근에는 대부분의 SVS 모델링이 TTS 모델의 **NN 구조에서 영감**을 받아서 제작되었음
        - 하지만 대부분의 제안 방법에서는 **timbre와 pitch representation을 분리**하기 위해서 linguistic and note information이 각각의 모델에서 **independently**하게 그리고 **unsupervised**한 방식으로 입력됨
        - 이런 방식은 natural singing voice를 생성하는데에 한계가 있는데, 이는 timbre와 pitch가 각각 **independent하다는 가정**하에 모델링이 된것이기 때문
    
    → timbre 와 pitch는 **interdepence**한데 이걸 **반영해주지 못하니까 문제**라는 것
    
- **singer 의 timbre 특성은 voice source이랑 vocal tract에 따라서 달라**지며, natural singing voice를 모델링하기 위해서는 **SVS가 timbre와 pitch간의 interdependence를 고려해서 모델링**되어야함
    - 최근에는 대부분의 SVS 모델링이 TTS 모델의 **NN 구조에서 영감**을 받아서 제작되었음
        - 하지만 대부분의 제안 방법에서는 **timbre와 pitch representation을 분리**하기 위해서 linguistic and note information이 각각의 모델에서 **independently**하게 그리고 **unsupervised**한 방식으로 입력됨
        - 이런 방식은 natural singing voice를 생성하는데에 한계가 있는데, 이는 timbre와 pitch가 각각 **independent하다는 가정**하에 모델링이 된것이기 때문
    
    → timbre 와 pitch는 **interdepence**한데 이걸 **반영해주지 못하니까 문제**라는 것
    
- 본 연구에서는 SVS model 기반의 a**dversarial multi task learning 을 제안**하며 해당 모델은 **timbre 와 pitch representation을 구별해 줄 수 있는 모델**임
    - **main task는 mel spectrogram을 예측**하는 것으로, **보조적인 task가 timbre와 ptich의 feature을 예측**하는 것
    - 제안하는 모델은 2가지 구간을 통해서 훈련되었으며 **discriminator을 통한 adversarial training**이 적용되었음
        - **첫번째 구간**에서는 **auxiliary task에 pre train**되었으며, **해당 pre training은 두개의 decoder이 timbre와 pitch를 각각 represent**할 수 있도록 했음
        - **두번째 구간**에서는 **메인과 보조적인 task에 대해서 jointly training**되었음.
            - **즉 각각의 decoder들을 통해서 timbre와 pitch의 mel spectrogram이 구분**되었으며, 이는 **final mel spectrogram에 통합되어 적용**되었음

# 2 Related Works

- Multi-Task Learning 은 speech application에 광범위하게 사용되었음
    - speech enhancement, recognition and synthesis
- multi task learning(MTL)은 학습 방법인데 multi ple task를 jointly 학습하면서 관련된 정보를 공유함으로서 모델의 성능을 높여 주었음
- 본 연구에서는 MTL을 SVS task에 적용하는데, WORLD vocoder이랑 neural vocoder의 feature prediction을 jointly learning하도록 설계함
- Conditional GAN framework을 통해서 multi singer SVS mdoel의 singer embedding을 생성함

# 3 Proposed Methods

- Architecture overview
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1uZ9hmGuB6n0rdSB0jC8H71fxilF3aP4f){: width="90%" height="90%"}{: .center}
    
    - Fig1- a) single task SVS model / Fig1 -b) multi task SVS model
    - a, b 모두 두개의 encoder과 두개의 decoder 그리고 mel predictor을 포함하고 잇음
    - a에서는 phoneme과 note pitch들이 독립적으로 모델링됨
    - b에서는 encoder의 output이 timbre와 pitch decoder으로 입력되기 전에 통합됨
        - 이렇게 함으로서 각각의 특성을 모델에 좀 더 반영해 줌
            - PREDICTED TIMBRE와 PITCH representation이 mel predictor에서 통합되고 final mel spectrogram을 예측함
        - mel spectrogram과 auxiliary feature들은 discriminator을 통해서 adversarial하게 훈련됨
            - auxilairy feature : Pitch, Timbre

## 3.1 Embeddings

- 이전 한국어 선행 SVS 연구들에서와 같이 본 연구에서는 아래의 이점을 사용함
    - 가사 속의 하나의 음절이 MIDI(Musical Instrument Digital Interface) 의 하나의 note에 매칭됨
- **임베딩 방법**
    1. 한국어 grapheme-to-phoneme 알고리즘을 통해서 **한국어 음절을 phoneme 시퀀스로 분해**해줌
    2. MIDI not sequence 의 시작 시간, 지속 시간 그리고 pitch 정보를 사용해서 각각의 **phoneme에 해당하는 frame과 phonmeme을 연결**시켜서 **frame level의 phoneme sequence를 획득**함
    3. **한국어 음절이 onset ,nucleus, optional conda 를 일반적으로 포함**하고 있기 때문에 onset과 coda를 첫번째와 마지막 세개의 frame에 각각 할당해주고, 나머지 frame을 nucleus에 할당해줌.
        1. onset : 초성
        2. coda : 종성
        3. nucleus : 중성
    4.  input sequence는  **학습가능한 lookup table을 통해서 D차원의 dense vector으로 임베딩** 됨. ( ET, EP)
        1. ET : 임베딩된 phoneme sequence
        2. EP : 임베딩된 note sequence
    5. ET와 EP는 phoneme encoder과 note encoder 에 각각 통과되고, 이후에 singer embedding vector ES으로 통합됨
        
        → 이것도 학습가능한 lookup table 틀 통해서 생성됨
        

## 3.2 Timbre and pitch decoders

- **모델의 두개의 decoder은 conformer block을 여러개 쌓아서 timbre 와 pitch representation들을 예측하고자 했음**
- 두개의 decoder이 encoder으로부터 받은 timbre 와 pitch의 개별적인 representation을 unsupervised한 방식으로 구별해 내지 못하기 때문에 **MTL 접근 방식이 소개되었음**
- explicitly하게 timbre와 pitch feature을 예측하는 auxiliary task을 통해서 두개의 decoder network은 timbre 와 pitch representation을 학습하기 위해서 공유되었음
    - timbre representation HT는 shared timbre decoder의 output임
        - HT는 Fully connected layer과 mel predictor에 input됨
            - FC layer은 timebre feature(MGC, BAP, V/UV flag들 같은거)을 예측하게 됨
    - pitch representation HP는 shared pitch encoder의 output임
        - HP도 mel predictor와 logF0을 예측하는 fully connected layer을 통과함
        - trianing에서는 groundtruth인 V/UV flag이 voiced section의 logF0을 훈련시키는 데에 사용되었음
        - 따라서 loss pitch에대한 loss 값은 아래와 같이 계산됨
            
            ![Untitled](http://drive.google.com/uc?export=view&id=15pbH7sm53Uda9KRjeSEKuATd7L2Y5AeC){: width="60%" height="60%"}{: .center}
            
            - 동그라미 기호는 element wise multiplication
            - p : target log F0
            - p^ : predicted F0
            - v : ground truth V/UV flag
            
            → 즉 예측된 F0과 V/UV flag의 정답을 element wise하게 곱해주고 p와 같이 L1 loss 함수에 입력해줌
            
        - auxiliary feature에 대한 loss값은 아래와 같이 계산됨
            
            ![Untitled](http://drive.google.com/uc?export=view&id=1cO8E2pIp9pfLwnzcAJlp6qvBRCcSEBJU){: width="60%" height="60%"}{: .center}
            
            - Lmgc : MGC의 Loss
            - Lbap : BAP의 Loss
            - Lvuv : VUV의 Loss
            - Lvuv만 binary cross entropy를 사용하고 다른 loss들은 다 L1 loss 함수를 사용함
            - wm,wb,wv,wp는 각각의 loss에 대한 스칼라 가중치 값임

## 3.3 Mel predictor

- 아래 그림의 c 파트에서 볼 수 있듯 timbre(Ht)와 pitch representation(Hp)은 log scale의 mel spectrogram을 예측하기 위해서 사용됨.
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1VUoYvBKtn9svZyMqaazYqETj-4YrRpav){: width="80%" height="80%"}{: .center}
    
    - 차원 축소를 위해서 Ht와 Hp는 각각 FC(Fully Connected) 레이어를 통과함
    - sigmoid를 마지막에 적용시켜주어서 output이 visible하고 interpretable 한 representation(Mt, Mp)이 되도록 함
    
    → **연구진들은 선행 실험에서Mt는 spectral envelope과 같고 Mp가 pitch harmonic과 동일하다는 것을 밝혀냈음**
    
    → source filter theroy에 기반하여, 선형적인 spectral domain에서는 pitch harmonic이 spetral envelope에 의하여  증폭됨
    
    → 하지만 본 연구는 log scale이 적용된 mel spectrogram을 다루기 때문에 multiplication은 summation operation으로 대체되었음(log계산에서 summation이 multiplication이니까)
    
    →따라서 Mt와 Mp는 summation되어서 convNN을 통과하게 되어 최종적인 mel spectrogram을 예측하게 됨 
    
    - source filter modeling이 mel spectrogram 영역에 적용되기 때문에 mel predictor의 posnet은 실제 mel spectorgram M과 가까운 예측 coarse mel spectrogram을 생성하게 됨
    - 본 연구에서 사용된 mel spectrogram의 loss는 아래와 같음
        
        ![Untitled](http://drive.google.com/uc?export=view&id=1uVpxBtPYC_PUwXJyyW6fpxWvn5cROHus){: width="60%" height="60%"}{: .center}
        
        - MT : timbre representation과 관련있는 representation 값
        - MP : pitch representation과 관련있는 representation값
        
        → 이 둘을 더하는게 souce filter 이론에 기반한 것
        

## 3.4 Discriminators

- singing voice의 perceptual quality를 향상시키기 위해서 conditional GAN framework을 제안했음
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1_9EUK3H0Sh_PE9KN647vgNEHjrd6_UNF){: width="60%" height="60%"}{: .center}
    
    - 선행연구 [26] 에서는 GAN framework을 mel spectrogram의 성능을 향상시키기 위해서 사용했지만, 본 연구에서는 mel spectrogram, MGC, logF0에 대한 세가지의 discriminator을 사용했음
        - 이를 통해서 timbre와 pitch representation을 구별하고자 햇음
    - 연구진들은 auxiliary feature을 reconstruction loss를 통해서 학습하는 것은 timbre와 pitch representation을 구분하는데에 충분하지 않다고 주장함. 더 나아가서 multi singer model에서의 MGC 와 LogF0은 높은 diversity들을 가지고 잇기 때문에 reconstruction loss 기반의 training은 MGC 와 logF0 예측 능력을 감쇠시킬 수 있음
        - 이러한 방식은 Ht와 Hp을 구별하지 못하게 entangled시키고 성능 저하를 야기함
    - 본 연구에서는 least squares loss function과 추가적인 feature matching loss를 adversarial training에서 사용함
        
        ![Untitled](http://drive.google.com/uc?export=view&id=1RSV_LtUvUguD8pgz1kWZeKx_HalrfCff){: width="80%" height="80%"}{: .center}
        
        - x → target feature
        - x^- > generated feature
        - s → singer embedding
        - SF = {m,t,p}은 mel spectrogram, MGC, LogF0에 대한 index set을 포함하고 있음
        - L→각 discriminator에 포함되는 전체 layer의 개수
    - generator의 전체적인 loss function은 아래와 같음
        
        ![Untitled](http://drive.google.com/uc?export=view&id=1mJcBagpdzwloeDPCGGahd3I6gdpauuON){: width="80%" height="80%"}{: .center}
        
        - pre training에서는 Lmel을 제외하고 SF={t,p} 을 통해서 timbre와 pitch representation을 구분함

# 4 Experiments

## 4.2 Results

### 4.2.1 Listening test

- MOS(Mean Opinion Socres)를 listening test에 대하여 적용함
- pass

### 4.2.2 Effect of adversarial training for auxilary task

- 아래 그림의 ㄱ은 mel spectrogram을 사용해서 adversarial training을 한 결과
- 아래 그림의 ㄴ 은 MGC와 log F0을 mel spectrogram과 같이 사용해서 adversarial training한 결과
- ㄴ의 b,c에서 확인할 수 있듯 spectral envelope과 pitch harmonic이 구분된것을 확인할 수 있음
    - ㄱ의 b,c에서 동그라미친 부분과 ㄴ의 b,c의 동그라미친 부분을 비교해보면 ㄴ이 훨씬 진동이 잘 보인다는 것을 통해서 해당 사실을 확인함
        
        ![Untitled](http://drive.google.com/uc?export=view&id=1C5df1HOfEePwRjjYOfjAu6mPnktS6fiV){: width="80%" height="80%"}{: .center}
        
- 정량적 평가에서도 좋은 평가를 받아냄
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1H9YnNpy--OaJVBlMYK2qd7DoTAi8mu-Q){: width="80%" height="80%"}{: .center}
    

# 5 Conclusions

- adversaial MTL 기반 SVS 을 제안하여 timbre와 pitch representation 을 구분함
- 제안된 방법론을 통한 결과는 single task SVS model보다 더 좋은 성능을 보임을 확인함


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