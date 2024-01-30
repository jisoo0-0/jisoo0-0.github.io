---
layout: post
categories: 논문리뷰
tags: audio, TTS
comments: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] FluentTTS: Text-dependent Fine-grained Style Control for Multi-style TTS."
---

출처
> Kim, Changhwan, et al. "FluentTTS: Text-dependent Fine-grained Style Control for Multi-style TTS." *Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH*. Vol. 2022. 2022.

Copyright of figures and other materials in the paper belongs to original authors.


# 0 Abstract

- 본 연구에서는 **TTS의 local prosodic variation을 유연하게 control하기 위한 방법을 제안**함
    - local prosodic variation이란 local한 음율적 변동임
- 합성된 음성에서 expressiveness를 제공하기 위해서 전통적인 TTS 모델은 utterance wise한 global style embedding을 사용함
    - utterance wise global style embedding은 frame level embedding을 time 축을 기반으로 compress 시켜서 얻어짐
    - utterance 란 발화를 의미하고 본 연구에서는 speech그 자첼 의미하는거 같음
    - **하지만 utterance wise global feature은 단어 수준의 local 한 feature에 대한 특성을 충분히 표현하지 못한다는 한계점이 있음. 따라서 해당 feature을 통해서 fine scale에서 prosody를 조절하는 것은 적절하지 않음**
        - prosody : 운율
- multi style TTS model에서는 local prosody를 control하는 것이 아주 중요한데 , **이는  local prosody가 one-to-many mapping 후보들중에서 올바른 text to speech pair을 선택하는데에 주용한 역할을 하기 때문임**
    - one to many mapping 문제
        - input text으로부터 다양한 ouput style option이 존재함
        - 본 연구에서는 global한 style embedding을 사용해서 이를 해결하고자 함
        - global style embedding은 target speaker의 identity와 prosodic 정보가 포함되어 있음
- local prosodic 특징들을 contextural information에 대하여 명시적으로 표현하기 위해서 본 연구에서는 F0(Fudamental frequency)를 예측하는 모듈을 제안함
    - 더 나아가서 **multi style encoder을 사용한 multi style embedding을 estimate**하고자 함
        - multi style encoder은 global utterance wise embedding과 local F0 embedding을 input으로 받음
    
    → **F0을 local style 을 representing하기 위해서 사용**
    

# 1 Introduction

- 대부분의 Neural TTS model들은acoustic feature들을 예측하고, 예측된 acoustic feature을 기반으로 합성된 음성 waveform을 neural vocoder으로 생성하고자 했음
    - acoustic feature : mel spectrogram, latent space embeddings from input phonetic features
    - 인간같은 speech 신호를 생성하기 위해서 neural TTS system들은 multiple speaker 의 identity 특성을 support할 수 있어야 하고, prosody related 정보를 사용해서 표현의 강도를 조절해야함
        - 표현의 강도란 말하기 속도나 감정이나 intonation(억양)
    - 하지만 이런 정보들을 통합하는 것이 쉽지 않은데 .. output 에대한 엄청나게 다양한 style option이 존재하기 때문임
        - ⇒ one to many mapping problem
            - one to many mapping problem을 해결하기 위한 가장 효과적인 방법은 발화자의 identity나 prosody related정보와 같은 추가적인 conditional embeding을사용하는 것임.
    - multi speaker TTS setting에서는 발화자의 identity 정보가 one hot vector나 발화자 embedding으로 표현될 수 있음
- prosodic variation을 제공하기 위해서 이전 연구들은 prosodic information을 예측하고자 했음
    - 이는 전체 입력 발화에 대한 prosodic information이었으며,  시간 축을 기반으로 acoustic feature들을 처리함으로서 얻어진 global 정보였음
    - 추출된 conditional embeding은 global style embedding이라고 불림
        - global style embedding과 text embedding을 더해서 decoding stage의 text related information의 input으로 사용함
    - 하지만 위와 같은 세팅에서는 같은 global style embedding이 각각의 text symbol에 대해서 적용되었으며 decoder은 text embeding에만 의존하여 local한 prosodic variation을 표현하고자 했음.
        - **즉, global style embedding은 전체적인 prosodic feature을 대변하지,  local한 prosody information을 decoder에게 명시적으로 전달해주지는 못함**
- multi sytle TTS system에서는 multi speaker과 multi emotion이 존재하고 이로인해서 prosodic style이 아주 급변할 수 있게 됨. 따라서 text symbol과 acoustic frame간의 적절한 time alignment이나 pair을 찾는 것이 어려울 수 있음.
- **본 연구에서는 local feature control 방안은 제안하며 좀 더 fine graine된 style embedding을 얻고자 함**
    - **TTS에서 fine grained style embedding은 local prosodic variation을 향상시킬 수 있음**
- 본 연구에서는 다양한 음향적인 feature들 중에 frame level fundamental frequency(F0) 을 local style을 대변하는 지표로 사용함
    - F0값의 예측 복잡도를 최소화하기 위해서 각각의 emotion type에 대한 발화에 대하여 F0 값을 각각의 speaker 의 평균과 표준편차를 사용해서 정규화하여 표준 정규 분포를 갖도록 했음
    - 입력 발화의 F0 값이 spectral acoustic feature frame과 같기 때문에 본 연구에서는 internal aligner을 사용해서 timing ingormation과 해당하는 text symbol을 합성해주었음
- 본 연구에서 제안된 모델의 간략한 training / inference process
    - training시 phoneme level freference F0 값을 internal aligner과 사용해서 F0 embedding을 생성해냄
        - 생성된 F0 embedding은 global style embedding과 나중에 concat됨
        - 동시에 모델은 conditional layer normalization을 통해서 global style embedding을 생성함
            - 모델은 global style embedding을 사용해서 input text symbol에대한 F0값을 예측하고자함
    - inference시 모델은 예측된 F0값을 통해서 F0 embedding을 생성해 내고, 이를 local prosodic variation을 조절하는데에 사용함
- **본 연구의 의의**
    - content dependent fine grained style feature을 생성해 내는 neural multi style TTS 모델을 제안함
    - 명시적으로 local prosodic variation을 prosody가 급변하는 경우에도 조절할 수 있음
    - naturalness, stable alignment, local prosodic variation의 baseline을 뛰어넘는 성능을 보여줌

# 2 Related Work

- transformer TTS
    - transformer의 multi head attention이 병렬적으로 계산되기 때문에 기존의 RNN기반 모델들보다 훨씬 빠른 속도를 지님
- multi speaker capability를 적용하기 위해서 Mulitspeech 선행 연구에서 여러개의 module을 추가하여 text와 speech간의 timing instant 정확도를 증진하고자 했음
    - 해당 연구는 text embedding에 positional embedding 적용하기 전 단계에서 text embedding에 layer norm을 사용하여 positional information을 보다 보존하고자 했음
    - 보다 안정적인 convergence를 위해서 encoder decoder attention module에 diagonal constraint를 적용하기도 함
    - Multispeech 선행연구는 multi speaker 시나리오에서의 one to many mapping 문제를 해결할 수 있었음
- multi style 시나리오에서는 speaker의 identy 뿐만 아니라 TTS model에게 prosody related 정보를 제공하는 것이 중요함.
    - Skerry-Ryan et al[11] 선행 연구에서는 2D conv layer 을 쌓아서 만든 reference encoder을 선보였는데 이는 local context를 capture할 수 있었음. 또한 GRU를 통해서 final hidden state에서의 representative한 prosodic feature들을 얻을 수 있었음
- prosody 를 global scale 뿐만아니라 fine scale하기 위해서 여러가지 연구들이 진행되었음
    - Lei et al[15] 은 pretrained ranking function을 Tacotron 기반의 구조에 적용시켰음
        - 해당 function은 neutral speech을 기반으로 emotional 한 강도의 특징을 estimate함
    - Fastpitch[19] 는 pitch predictor을 사용해서 명백하게 F0을 control할 수 있는 방안을 제안함
        - pitch predictor은 text의 평균적인 pitch value를 예측하고자 함
            - 이를 위하여 external aligner으로부터 추출된 ground truth을 targeting함
- 그동안의 연구 덕분에 TTS의 성능은 확연히 좋아졌지만, external 모듈을 사용하지 않고 SOTA TTS 모델을 제안하는 것이 next challenge 였음
    - Badlani et al[20] 은 soft alignment를 사용해서 text 와 speech사이의 probabilistic 분포를 학습할 수 있는 internal aligner 을 제안함
        - 또한, hard alignment 을 soft alignment 와 같은 특성을 가지도록 하는 적절한 path를 찾고자 했음
            - hard alignment는 0과 1의 값을 포함하고 있어서 input text embedding이 acoustic feature frame과 같은 길이가 되도록 up sampling하는 데에 사용될 수 있음
    - soft alignment vs hard alignment
        - soft alignment는 사람이 따로 target을 알려주지 않음
        - hard alignment는 사람이 따로 target을 알려줌
- 본 연구에서는
    1. multi style TTS model이 content-dependent gine-grained prosody를 외부 모듈 없이도 조절가능하도록 함
    2. F0 값을 각 text symbol에 대하여 생성하기 위해서 internal aligner으로부터 획득 가능한 hard alignment output을 사용함
    3. conditional layer norm을 통한 global style information을 사용해서 각 text symbol에 대한 F0값을 예측함

# 3 Proposed Method

- 모델의 전반적인 구조는 아래 figure 과 같음
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/25f9a086-561c-4641-b296-e1efac815de3){: width="90%" height="90%"}{: .center}
    
    - 베이스라인을 Multi speech으로 설정하고, repference encoder[11] 을 통해서 emotion과 관련된 정보를 추출하고자 했음
    - 해당 모델들과 본 연구에서 제안한 모델들의 차이는 multi style generation 부분의 auxiliary 모듈 부분임
    - 그동안은 text embedding과 mel spectrogram 간의 binary로 구성된 hard alignment를 생성해 냈음
        - hard alignment를 통해 각각의 text symbol에 대한 frame의 개수를 정의할 수 있음
        - hard alignment 정보를 통해서 phoneme level F0를 구할 수 있음
            - 각각의 text symbol의 지속시간에 대해서 정규화된 reference F0의 평균값을 취함
    - 본 연구에서 제안된 모델은 아래와 같은 processing step을 따름
        1. speaker을 생성하고 speaker encoder과 reference encoder으로부터 emotion embedding을 각각 softmax를 취해서 들고옴
        2. 둘은 concat되어서 fully connected layer를 통과하는데 이 과정을 통해서 conv는 둘의 정보는 합쳐주고 차원은 축소해줌
            1. fully connected layer을 통과해서 나온 값은 global style embedding임
        3. F0 predictor을 통해서 F0값을 구하고 이 값을 conv1d를 통해 임베딩 시켜줌
        4. F0 embedding값과 global style embedding 값을 concat해서 multi style embeding을 multi style encoder에서 생성해냄
        5. decoder이 text와 multi style embedding을 사용해서 mel spectrogram을 생성해냄
    - 제안된 모델의 훈련 목표는 input text으로부터 phoneme level의 F0을 얻는 것임
        - 연구진들은 F0 predictor을 통해서 적절히 정규화된 F0값을 얻고자 했음 \
        - F0 embedding 값은 F0값을 Conv1D layer을 통과시켜서 얻을 수 있음

## 3.1 F0 generation

- Raw한 F0값이 speaker identity와 emotion type에 따라서 엄청나게 큰 범위에서 움직이기 때문에 model이 raw한 F0값을 directly 하게 예측하는 것은 어려움
- 따라서 F0값을 평균과 표준편차를 통해서 정규화하고 이를 예측하고자함
- speech signal value으로부터 reference F0 값을 추출하기 위해서 WORLD vocoder을 사용함

### 3.1.1 Text-dependent average F0

- input text으로부터 F0값을 학습하고 예측하기 위해서는 **input text에 대하여 F0값의 길이는 동일**해야함
- t**arget F0 값이 speech signal으로부터 추출**되고, 해당 값이 **모든 anaysis frame에 대해서 획득**되니까 **target F0값의 길이**는 **mel spectrogram frame의 길이와 동일**하게 됨
    - **하지만 해당하는 F0값의 길이가 input text 길이와는 다르다는 문제점이 발생함**
    - 우리는 average F0값을 off line external aligner 을 통해서 각각의 text interval에 대해서 계산할 수 있음. 하지만 만약 F0 estimation process에 오류가 있다면 time alignment 문제를 발생시게 됨.
- 따라서 본 연구에서는 internal aligner을 통해서 frame level F0 값과 이에 상응하는 text symbol을 성공적으로 합성시켜줌
    - internal aligner 가 hard alignment 를 output으로 생성해 내고, 이를 통해서 text symbol과 mel spectro gram의 프레임을 1:1으로 매칭시켜줌
    - 해당 정보를 이용해서 각각의 phonetic symbol에 상응하는 reference average f0 값을 얻을 수 있게 되고, 본 연구에서는 **phoneme level F0값을 content dependent fine grained style feature으로 사용함**

### 3.1.2 F0 prediction in the inference stage

- reference F0 값이 inference stage에서 사용될 수 없기 때문에 모델으로 하여금 해당 값을 예측하도록 해야함
- 아래 그림은 F0 predictor의 구성을 나타냄
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/4c31277b-41dd-4b04-b67f-a5e6811df7bb){: width="80%" height="80%"}{: .center}
    
    - 이 구성은 Fastspeech2 의 pitch predictor과 Adaspeech의 conditional layer normalization으로부터 영감을 받아서 제작되었음
    - input text related embedding으로부터 f0 값을 보다 정확하게 예측하기 위해서 figure과 같이 3개의 layer을 쌓아서 구성함
    - Conditional layer norm을 통해서 발화 wise global style embedding을 제공하여 F0 predictor이 해당 발화들에 기반하여 생성되도록 했음
    - additonal fully connected layer을 통과한 뒤에 F0 predictor은 정규화된 F0값을 예측하게 됨
    - F0 값이 speaker의 identity와 emotion type과 강하게 관련이 있기 때문에 F0을 사용하여 speech sample을 생성하는 것은 make sence함

## 3.2 Multi style encoder

- 아래 그림은 multi style encoder에 관한 전반적인 구조를 나타냄
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/0aff6823-022b-4bdc-9da2-2ece1f154a94){: width="50%" height="50%"}{: .center}
    
    - baseline 구조는 F0 predictor과 동일하지만 stack되지 않았다는 점과 last layer에 softsign function이 포함되지 않았다는 점이 다름
        - softsing function은 output embedding의 dynamic range을 restrict 해줌
    - multi style encoder은 global style embedding과 local F0 embedding을 concat해서  input으로 받음
    - global style embedding과 local embedding의 특성이 급변할 수 있기 때문에 multi style encoder은 input embedding들의 dynamic한 특성을 다룰 수 잇어야함
    - global style embedding이 text sequence에 따라서 expanding되는 것과 다르게, multi style encoder의 output은 F0 embedding의 type에 따라서 vary하게 됨
    - F0 embedding이 text dependent average F0으로부터 생성되기 때문에 multi style embedding이 text dependent 한 local prosodic information을 포함하고 있다고 볼 수 있음
    - global style embedding이 전체적인 입력 발화에 대한 prosodic variation을 포함하고 있기는 하지만, model이 local prosodic variation을 reconstruct하는 것은 힘들 수 있음
    - 하지만 제안된 모델은 global과 local prosodic variation을 합치면서 이를 해결했음
        - global→ global style embedding에서 획득, local → f0 embedding을 통해서 획득

## 3.3 Dynamic level F0 control

- 제안된 모델은 발화의 prosodic변화를 control 하고자 함.
- model이 입력 text symbol에 대응하는 F0 값을 예측하기 때문에, speech signal을 유연하게 합성해서 F0 modification이나 desired prosodic variation을 가질 수 있도록 함.
    - 예를 들어서 만약 F0을 다른 단어에 해당하는 것으로 바꾸고 싶다면, 해당 단어가 입력 발화에 어디에 위치하는지를 확인하고 해당하는 F0값을 우리가 원하는 값으로 바꾸면 됨

## 3.4 Training objective

- 본 연구에서 사용된 최종적인 loss값은 아래 식과 같음
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/dd9fa742-1d93-4fdb-b839-68b1b96773a3){: width="80%" height="80%"}{: .center}
    
    - LTTS : TTS training의 손실값
    - LIA : internal aligner module에 대한 손실값
    - LF0 : F0 predictor module에 대한 손실값
    - 본 연구에서는 F0 loss를 20k step이 되기 전까지는 사용하지 않고 global style embedding을 사용함
        - 이건 20k 이후에 F0 loss를 사용하는 것이 training process를 안정적이게 하기 때문임
- LTTS는 아래와 같이 정의됨
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/5fdbb72b-2fc4-44f7-9300-e90325b7a1c6){: width="80%" height="80%"}{: .center}
    
    - LREC : mel spectrogram reconstruction에 대한L1 손실값
    - LBCE : stop token에 대한 binary cross entropy 손실값
    - Lguide : encoder decoder attention에서의 stable alignment을 위한 guided attention loss를 의미함
    - Lemo : reference encoder에 대한 emotion 분류 손실값
- internal aligner module을 학습하기 위해서 training process의 초기에 CTC loss를 적용하고, 10k step마다 KL divergence loss를 아래와 같이 포함시켜주었음
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/29e97583-7171-42cc-a191-245f45f65ea1){: width="80%" height="80%"}{: .center}
    

# 5 Conclusion

- multi style TTS에 대하여 text dependent한 fine grained style control method 를 제안함
- phoneme level F0 value를 internal aligner을 통해서 생성해 내어서 fine grained prosody를 조절할 수 있도록 했음
- local 한 정보를 반영해주었음


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