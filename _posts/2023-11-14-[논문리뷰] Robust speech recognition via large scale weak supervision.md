---
layout: post
categories: 논문리뷰
tags: ASR, weak-supervision
comments: true
math: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] Robust speech recognition via large-scale weak supervision"
---


**논문 및 사진 출처**
>Radford, Alec, et al. "Robust speech recognition via large-scale weak supervision." International Conference on Machine Learning. PMLR, 2023.


# Abstract

- speech procdessing의 capability들에 관한 연구
    - large amounts of transcripts of audio in the internet 사용

# Introduction

- Wav2Vec2.0과 같은 방법이 raw audio 으로부터 바로 학습이 되기 대문에, 대량의 데이터셋을 사용할 수 있었음.
- 특히 low-data setting에서는 이러한 standard benchmark에 fine-tuning하면 성능을 올릴 수 있었음
- 이러한 pre-trained audio encoder은 speech에 있는 high quality representation을 학습할 수 있지만, purely unsupervised한 방식을 사용하기 때문에 decoder가 이러한 representation을 output에 맵핑하려면 fine-tuning 작업이 필요했음
- 그런데 이러한 fine-tuning 작업은 복잡하고 숙련된 사람을 요하기 때문에 한계가 있음
    - 추가적으로 머신러닝 방법들은 training dataset 속의 pattern들에 adapt되는데 이러한 pattern들이다른 dataset에는 generalize되기 힘듦
- 선행연구에서 증명되었듯이 supervised 방법으로 여러 dataset/domain에 pre-trained된 speech recognition system들은 robust하고 generalization 능력이 높다는것을 알 수 있음.
    - 하지만, 데이터 부족으로 한계가 있음.
        - 5140 시 labeled 간 vs 1000000 시간의 unlabeled 데이터
- supervised dataset의 양이 부족하다는 것을 기반으로 speech recognition의 큰 데이터셋이 제작되었음.
    - 하지만 이러한 new dataset도 unsupervised work에서 사용되는 데이터보다 훨씬 작음
- 본 연구에서는 이러한 gap을 줄이고자 했음.
    - 더 나아가서 english-only speech recognition 을 넘어서 multilingual , multitask에도 적용해보고자 했음.

# 2 Approach

## 2.1 Data processing

- While diversity in audio quality can help train a model to be
robust, diversity in transcript quality is not similarly beneficial
    - To address this, we developed several automated filtering methods to improve transcript quality.
    - → web 기반으로 데이터를 수집하면 오디오 퀄리티의 다양성 및 자막 퀄리티의 다양성을 얻을 수 있음. 오디오에 대한 다양성은 모델을 로버스트하게 하지만 자막은 아니기 때문에 이를 필터링하는 것을 적용했다는 것
- 인터넷에 있는 자막들은 사람들이 적은거보다는 기계로 인해서 생성된 것이 많음. 그래서 이와 같은 것(머신이 만든 데이터)를 제거하기 위해서 휴리스틱한 방법을 적용했음.
- 추가적으로 audio language detector을 사용해서 오디오와 자막이 매칭되지 않으면 해당 pair을 제외시켰음.
- low quality 를 보이는 set도 제외시킴

## 2.2 Model

- scale reliably가 있기 때문에 transformer을 선택함
- 모든 오디오는 16,000 헤르츠로 리셈플링 되고, mel spectrogram representation으로 전처리됨
- feature norm 진행
- 인코더는 input representation을 2개의 conv layer + GELU activation으로 process함
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/d2260cdb-703a-4418-9539-96d33e70cf58){: width="40%" height="40%"}{: .center}
    
    
    - sinusodial position embedding이 더해지고, encoder output에는 pre-actiation residual block과 final layer norm이 적용됨
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/0c83459b-244c-4e06-9882-afc2f9ed6402){: width="40%" height="40%"}{: .center}
    
        
- 디코더는 learned position embedding 을 사용하고 tied input-output token representation들을 사용함

## 2.3 Multitask format

- 어느 단어가 speech에 있는지 예측하는 것 말고도 여러가지 additional 한 component들이 존재함.
    - voice activity detection
    - speaker diarization
    - inverse text normalization
- 위와 같은 부가적인 component들은 주로 분리되어서 처리되는데, 이렇게 되면 core한 speech recognition model주변에 너무 상대적으로 복잡한 시스템이 구성되게 됨.
- 이러한 복잡성을 줄이기 위해서 본 연구의 저자들은 전체적인 speech pipeline을 처리할 수 있는 싱글 모델을 제안함.
- 같은 오디오 시그널을 기반으로 다양한 테스크를 처리할 수 있는데, 싱글 모델이 이렇게 one-to-many mapping 을 하려면 task specification이 필요함.
    - 본 연구에서는 simple format을 사용해서 모든 task를 decoder에 명시해주고자 함.
    - 연구에서 제안된 decoder이 audio-conditional LM 이기 때문에, model을 history of the text에 대한 condition을 학습하도록 하게함.
    - prediction의 begining 을 startoftranscript 토큰으로 설정했음.
- 우선 무슨 언어로 발화되었는지를 나타내는 토큰을 상요함.
    - 이러한 언어 타겟은 VoxLingua107 모델으로부터 나옴.
    - 만약 발화된 말이 없을 경우 , nospeech token이 사용됨.
    - no timestamps token을사용해서 timestamp들을 예측할건지 아닐지를 나타내줌.
        - timestamps 에측에서는 현재 오디오 세그먼트와 가장 관련있는 time을 예측하고, 모든 시간을 가장 가까운 20 밀리세컨즈로 양자화해서 Whisper model의 time resolution과 매칭될 수 있게 했음.
            
            ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/e6b99b3f-4c36-474f-bad8-e4420d6bb0ba)
    
            
        - no timestamps가 없다면 오디오와 시간축 상으로 딱딱 매치되지 않을 수 있음.
            - 그리고 오디오는 연속적인 데이터이기 때문에 어디까지 inference가 되었는지 모델에게 알려줘야지 그 이전 시점의 정보가 반영되지 않음. 반영될 필요도 없고
- 실험 결과에서 헷갈렸던 부분
    
    Inspection shows the majority of supposedly Welsh translation data is actually English audio with English captions where the English audio was mis-classified as Welsh by the language identification system, resulting in it being included as translation training data rather transcription data according to our dataset creation rules.
    
    → 이건 welsh language dataset에 단순히 english data가 많이 섞여있어서 문제라는게 아니라, dataset transcription data로 분류되어야하는 것이 translation data로 들어갔기 때문에 발생한 것.  
    
    → 처음에는 그냥 영어가 섞여서 nosiy해졌다는줄알고 이해가 잘 가지 않았는데 애초에 task 자체에 혼돈이 생기게끔 logic이 흐르는 거라서 해당 데이터가 노이즈라는 것을 납득함



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