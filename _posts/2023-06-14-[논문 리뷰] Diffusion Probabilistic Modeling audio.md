---
layout: post
categories: 논문리뷰
tags: audio, voice_conversion, voice_cloning
comments: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] A Unified System for Voice Cloning and Voice Conversion through Diffusion Probabilistic Modeling"
---

출처
>  Sadekova, Tasnima, et al. "A Unified System for Voice Cloning and Voice Conversion through Diffusion Probabilistic Modeling}}." *Proc. Interspeech 2022* (2022): 3003-3007.

Copyright of figures and other materials in the paper belongs to original authors.

# 부가정보

- Voice Cloing VS Voice Conversion
    - Voice cloning is **the creation of an artificial simulation of a person's voice**.
        - 출처 : [https://www.mdpi.com/2076-3417/13/5/3100](https://www.mdpi.com/2076-3417/13/5/3100)
    - Voice Conversion **the science of transforming one voice to sound like another person's voice without changing the linguistic content**
        - 출처 : [https://www.idrnd.ai/what-is-voice-cloning/](https://www.idrnd.ai/what-is-voice-cloning/)
    - 즉 cloning은 생성에 초점 conversion은 transform에 초점
    - 특히 cloning은 TTS 에 target 음성을 추가해서 음성을 생성하는데에 초점

# 0 Abstract

- Text to speech 와 voice conversion은 speach generation task의 보편적인 task임
- 본 연구에서는 voice cloning과 voice conversion의 새로운 방법론을 제안함

# 1 Introduction

- Voice Cloining task에서는 주로 multi speaker text to speech system에 새로운 voice를 adding하고자 함
    - neural TTS model은 주로 엄청나게 큰 transcribed data를 이용해서 학습을 요함
    - 이러한 제약 사항을 극복할 수 있는 가장 보편적인 방법은 multi speaker TTS system을 target speaker data에 fine tuing 하는 것.
        - 이를 주로 speaker adaptation이라고 부름
- voice cloining이 text to speech 기술에서 target speaker의 voice를 복사하는데에 필수적인 것에 반해, voice conversion의 다른 approach가 이를 achieve 할 수 있음
    - 이러한 접근 방법에서는 source 발화에 대한 언어적인 정보가 text가 아닌 speech으로부터 추출됨
    - any-to-any voice conversio model이 선호되는데, 이는 해당 모델은 source 와 target speaker이 training dataset에 포함되지 않더라도, source speech의 content을 유지하면서 target voice를 copying할 수 있음
- TTS와 voice conversion이 공통된 지점을 많이 공유하지만, 이를 같이 ㅅ용해서 multimodal system을 설계한 연구는 많지 않음
- 본연구에서는 새로운 hybrid system을 제안함
    - the text encoder
    - the mel encoder
    - the shared encoder
    
    → 전체적인 system은 Diffusion Probabilistic Model(DPM) 임
    
    - The whole system is essentially a diffusion probabilistic model (DPM) trying to convert speaker-independent acoustic features extracted either from text **by means of** the text encoder or from source spoken utterance **by means of** the mel encoder to the target melspectrogram **by** employing speaker-dependent score matching network which we call the decoder. → bybyby.. haha.. ha… ha…. 제발..
    - 정리하면 DPM은 speaker independent acoustic feature을 2개의 source으로부터 추출해냄
        - Text encoder을 사용해서 text에 대한 source 획득
        - Mel encoder을 사용해서 spoken utterance source 획득
            - speaker-dependent score matching network(=decoder)을 사용해서 target melspectrogram으로 변환
    - DPM은 다양한 speech 관련 task에서 좋은 성능을 보였기 때문에 model의 능력을 extend 시켜서 본 연구에서 사용할 것
    - 더 나아가서 제안된 모델의  speaker adaptation의 hybrid nature 이 untranscribed data에도 사용될 수 있음을 보였음
    

# 2 Multimodal System Description

- 선행연구 17과 같은 디퓨전 모델링 프레임워크를 따름
- forward diffusion은 any mel spectrogram X0을 normal random vraiable X1~N(X바, I) 으로 변환시킴
    - I 는 identity matrix이고 X바는 mel spectrogram의 평균적인 voice임
        - mel spectrogram은 mel encoder으로부터 predict되기 때문에 prior N(인코더(X0),I)는 speaker independent 하고 encoded speech X0의 linguistic한 content를 보존하게 됨
- speaker-conditional decoder으로부터 parameterized된 reverse diffusion은 연속적인 시간에서의  forward diffusion trajectory의 backward를 추정할 수 있도록 학습됨
- 결과적으로 well trained 된 decoder은 prior N에서 X^1을 sampling함. 해당 decoder로 parametrized된 reverse diffusion path를 적절한 numerical 방식으로 simulating해서 생성 모델을 가능하게 함.

## 2.1 Mel encoder

- Mel encoder 은 output mel spectrogram과  average voice mel spectrogram 간의 차이인 MSE를 줄이도록 학습됨
- mel feature의 ground truth는 LibriTTS의 Montreal Forced Aligner을 통해서 획득됨.
- LibriTTS을 통해서 input mel spectrogram X0의 각 phoneme에 해당하는 feature을 LibriTTS 데이터셋에서 평균화된 이 특정 phoneme에 해당하는 특징으로 대체함
- phoneme은 복수의 acoustic frame 동안 지속될 수 있기 때문에, 모든 phoneme은 single frame에 대응하는 평균 mel feature으로 표현되고, 특정 phoneme에 해당하는 모든 frame들은 align되어서 average voice mel spectrogrames 의 mel feature 값으로 대체됨.
- mel encoder은 pre net으로 구성되어 있고, 6개의 transformer block(with multi head self attention ) 과 final linear projection layer으로 구성되어 있음
- pre net은 3개의 1D conv layer와 fully connected layer으로 구성됨

## 2.2 Decoder

- mel decoder이 DPM prior N을 parametrizing하는 것을 학습한 후에, decoder의 parameter은 fix되고 decoder에 해당하는 reverse diffusion이 training됨.
- 본 연구에서는 Stochastic Differential Equation(SDE)을 통해서 DPM을 적용함
- forward X와 reverse X^의 diffusion process는 다음과 같음
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/a5e2587c-d09b-40a8-b598-29bbe4b0fa6d){: width="50%" height="50%"}{: .center}
    
    - 1번 식은 Forward SDE solver
    - 2번 식은 SDE-ML solver 즉 Decoder
        - 랜덤 variables X^1을 target mel spectrogram X^0으로 변환
    - Decoder의 경우인 score matching network 함수인 s세타로 파라미터 조정을 추가적으로 함
    - 베타 t는 noise 스케줄로 추가되는 잡음의 강도를 제어함
    - W는 각각 전진, 역확산 표준 브라운의 운동 방향을 나타냄
- target speaker의 identity를 캡쳐하기 위해서 speaker encoding network gt()를 **score matching network s세타**에 적용해서 두개의 network을 jointly하게 학습함
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/fa459abc-0302-429f-b9ed-1cdeca3d6ca0){: width="30%" height="30%"}{: .center}
    
    - score maching loss는 decoder을 통해서 만들어진 mel spectrogram과 text encoder이나 mel encoder을 통해서 만들어진 spectrogram의 값 차이에 관한 loss임
    - Y는 target speaker 에 대해서 계산된 reference mel spectrogram의 전체 궤적을 타나내며 이는 forward diffusion을 통해서 계산됨(1번식)
- decoder은 log likelihood의 data에 weighted variational lower bound 를 최대화 할 수 있도록 학습됨. score matching loss는 아래 식과 같음
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/8376b4a1-825e-41eb-86fa-409040797727){: width="50%" height="50%"}{: .center}
    
    - 해당 손실은 모델과 실제 데이터의 score을 일치시키고자 함.
    - 이를 통해서 모델은 실제 데이터의 특성과 분포를 보다 잘 모방할 수 있도록 학습됨
    - Xt는 다음 방정식을 통해서 획득됨
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/0033c55a-1ca6-4799-a642-1f7025d60d54){: width="50%" height="50%"}{: .center}
        
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/dbbf60da-5721-45ca-b419-2003f312f598){: width="80%" height="80%"}{: .center}
    
- decoder은 UNet 기반의 score matching network s세타를 포함하고 있음
    - decoder은 3*3Conv으로 mel spectrogram을 2D image으로 processing함
        - Conv 채널은 채널은 256,512,1024
    - speaker encoder gt(Y)는 target speaker embedding을 processing함
        - speaker encoder은 nosiy mel spectrogram Yt 와 broadcast concat됨
            - 이 과정에서 3*3 conv stack이 사용되며, channel은 128에서 512로 증가, average pooling이 사용됨
- generation 단계에서는 maximum likelihood reverse SDE(2)를 사용

## 2.3 Text Encoder

- 본 연구에서는 TEXT CONDER으로 Non attentive tacotron(NAT)을 변형해서 사용함
    - Non Attentive Tacotron
        - autoregressive TTS model
        - attention mechanism을통해서 중요한 텍스트 부분에 attention해서 audio signal을 생성함. NAT는 입력 텍스트를 순차적으로 처리하며 음성 신호를 생성함.
        - 
    - NAT는 Tacotron2의 변형 모델으로, attention model의 acoustic feature generator가 explicit한 duration predictor으로 변경되었음
- 본 연구에서는 NAT의 decoder 부분이 제거되었음
- text encoder은 일반적인 TTS model처럼 trainng 되었음. 다만, ground truth acoustic featrure 들이 mel spectrogram의 average voice라는 인공적인 값이라는것이 다름.
- target acoustic feature의 본성으로 인해서, 본 연구에서는 simple upsampling procedure을 적용함
    - 예측된 지속 시간에 따른 반복을 통해서 간단한 upsampling진행
- MSE loss를 통해서 학습됨

## 2.4 Operating modes

- 제안된 모델은 voice cloning과 voice conversion 모두를 perform가능함
    - decoder 앞에 오는 mel encoder은 voice conversion을 수행가능함
    - 디코더와 결합된 text encoder는 voice cloning 을 수행 가능함
- speaker adaptation은 decoedder을 fine tuning 하는작업을 포함하는데, text와 mel encoder들은speaker independent하도록 remain됨
    - 즉, speaker adaptation이 untranscribed data에 적용가능한데, 이는 decoder 학습시 target mel spectrogram과 average voice mel spectrogram만 필요하기 때문임

## 2.5 Related Models

- voice cloning과 voice conversion을 동시에 가능하게 한 최초의 모델은 NAUTILUS임
- 본 연구에서 제안한 모델과는 다르게 NAUTILUS는 mixed loss function을 통해서 jointly training되었음

# 3 Experiment

## 3.1 Comparison with multimodal systems

- 결과표
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/2b94af36-3b8b-47f1-9c14-6510bcdaae8e){: width="60%" height="60%"}{: .center}
    
    - table 1은 voice cloning과 voice conversion을 선행연구 13 모델과 비교
    - table 2는 voice cloning을 선행연구 14와 비교

## 3.2 Comparison with single task systems

- 결과표
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/fbf548a0-9353-43f1-824e-6b37881c8b76){: width="60%" height="60%"}{: .center}
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/30dd4f22-be72-44db-aa28-32105fbfbe0c){: width="60%" height="60%"}{: .center}
    
    - table 3,4 를 통해서 single task에서 time 관점에서 더 좋은 결과를 보임을 확인 가능함
    - table 5에서 제안된 모델이 domain shift에 좀 더 robust해서 realistic 한 condition에 좀더 stable한 것을 확인 가능함

## 3.3 Adaptation data requirements

- 결과표
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/a1b69a8c-dc01-4c33-beff-87c9a8cd976f){: width="60%" height="60%"}{: .center}
    
    - adaptation data의 영향이 speech 합성의 quality에 어떤 영향을 주는지 확인
    - 제안된 모델은 15초동안 adaptation 과정을 거친 것만으로도 좋은 성능을 보임을 확인

# 4 Conclusion

- Diffusion model을 사용한 새로운 multimodal system 제안
- speaker adaptation동안 본 system은 mel encoder을 사용해서 adaptation audio에 대한 transcription이 필요가 없었음
- voice cloning과 voice conversion 두 task에서 모두 좋은 성능을 보임
- 낮은 데이터 adaptation으로도 좋은 성능을 보임


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