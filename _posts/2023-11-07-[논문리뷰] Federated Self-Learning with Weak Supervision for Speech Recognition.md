---
layout: post
categories: 논문리뷰
tags: federated-learning, self-learning, weak-supervision, ASR
comments: true
math: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] Federated Self-Learning with Weak Supervision for Speech Recognition"
---


**논문 및 사진 출처**
>M. Rao et al., "Federated Self-Learning with Weak Supervision for Speech Recognition," ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10096983.

# Abstract

- ASR model with low-footprint에 대한 모델들은 conversational agent들의 edge devices에 적용되고 있음.
- 본 연구에서는 RNN-T ASR model 을 위한 federated continual incremental learning을 해결하고자 함.
    - 특히 privacy enhancing scheme on-device 방법을 사용하고, human transcripts이나 machine transcription에 대한 access를 하지 않음
- 저자들은 self-learning based 방법을 사용하는데 ASR model의 Exponenetial moving average을 통해서 업데이트 되는 paired teacher model을 사용함
- 더나아가서 feedback scores이나 NL understanding semantics 과 같은 possibly noisy weak supervision signals을 제안함
    - NL understanding semantic과 같은 경우에는 user behavior으로부터 결정됨

# 1 Introduction

- voice tech의 On-device deployment는 cloud의 reliable한 network connection 없이도 conversational agent들을 가능하게 함.
    - 이는 processing을 위해서 cloud으로 발화를 전송할 필요를 없애서 lower-latency responses을 가능하게 함.
- ASR이 on-device에 deployed되었을 때 모델은 specific acoustic 이나 linguistic content specific 에 대해서 adapted되어야하고, temporal adaptation도 진행되어야 함.
- 본 연구에서는 우리는 continually / incrementally updating ASR models 을 federated setting에서 체크하고자 함.
    - privacy-enhancing features 은 1) 발화가 cloud으로 전송되지 않는 경우 2) 오디오에 대한 영구적인 저장소가 요구되지 않는 경우 3) audio에 대한 human의 gt annotation이 필요없는 경우임
- Privacy-preserving machine learning은 privacy risk에서 좀 free한 상태로 user data를 학습할 수 있게 하는 learning 방법임
- federated learning(FL) 은 privacy-preserving learning framework중에 가장 유명한 방법중 하나로, model을 on-device에서 학습하게 함.
    - conversational AI에서 user와 agent가 여러번 교류하게 되는데, 이러한 interaction에서 request가 제대로 처리되었는지 확인해야함
- FL에서 participating device으로부터 여러 모델은 update되어서 central server에 매 round마다 aggregated됨.
- FL은 speech application에서 잘 작동하는 것이 증명되었지만, model을 scratch부터 학습하는 것을 목표로 했음
    - fine-tuning 하는 방식이 아님.
    - 더 나아가서 이전 연구들은 static data을 고려했음.
        - round마다 바뀌는게 아니라 고정된 것이기 때문에 static이라고 표현한듯?
- 본 연구에서는 FL setting을 고려하려고 하는데, 모델이 잘 학습될 수 있도록 초기화하는 것을 탐구하고자 함.
    - 이는 round가 바뀌어도 persisted하지 않음.
- Semi-supervised learning은 ASR 을 unlabelled data를 기반으로 진행됨.
- Unsupervised approaches such as data2vec or WavLM은 contrastive object function을 사용해서 speech model을 pretrainㅎ고 finetuning시킴.
- unlabelled data에 대한 수도라벨을 생성하기 위해서 강한 teacher model을 사용하는데, 이는 on-device learning시에는 자원 문제때문에 적용하기 힘듦
- nosiy student learning이나 iterative pseoudo-labelling approach들은 ASR model을 사용해서 self-label celan audio를 함.
    - 이 때, ASR model은 augmented version of the audio 에 대해서 trained된 애들임
    - 그런데 이때 모델의 신뢰도가 낮은 경우에는 audio 가 추가적으로 filtered 될 수 있음
- 본 연구에서는 hybrid HMM-DNN 와 connectionist temporal classification(CTC) ASR model이 paired teacher model이 student model의 exponential moving average을 통해서 updating 되는 방식을 사용함
    - 이러한 방법은 아직까지 RNN-T ASR model에 적용되지 않았음.
- 본 연구에서는 self-learning을 weak supervision과 통합함.
    - conversational agent에서는 multiple turns in a section의 user가 interact함
    - prior work에서 확인된 내용은, “later interaction을 통해 request가 정확하게 handled 되었는지를 알 수 있다는 것”
        - 만약 user가 request를 취소 혹은 반복한다면, dissatisfaction이 signalled됨
        - terminal request에 대한 semantic은 initial request에 대한 feedback을 통해 나타날 수 있음
        - 위와 같은 방법이 gt transcription은 아니지만, 이러한 시그널들을 통해서 ASR model을 업데이트 하고자 했음.
        - 추가적으로 suer가 feed back을 명시적으로 남길 수 있는데 이는 feed back score으로 반영됨.
        - 위와같은 arbitrary reward를 반영해주기 위해서  REINFORCE라는 선행 연구의 framework을 사용했음
- contributions
    - ASR model에 unlabelled audio 데이터를 사용해서 incremental update을 진행했음.
        - federated 방법, edge device에 사용
        - 성능도 잘나옴

# 2 Methods

## 2.1 RNN-T ASR model architecture

- RNN-T는 encoder, prediction network, joint network으로 구성됨.
- encoder은 acoustic model과 비슷한데, 이는 encoder이 acoustic input feature에 대한 시퀀스를 입력으로 받고, encoding된 hidden representation을 output하기 때문임.
- LM 에 대응되는 prediction network은 previous output label prediction을 받아서 corresponding hidden representation으로 mapping해줌
- joint network은 feed forward network으로 encoder와 prediction network 의 hidden represenation들을 다 받아와서 final output label probablities를 output함

## 2.2 Federated self-learning for ASR

- semi-supervised learning approache들은 주로 strong teacher model을 기반으로 audio data에 대한 자막을 생성함.
- on-device에서는 teacher모델과 같은 큰 모델은 작동되기 힘들 수 있음.
- 본 작업에서는 federated constraints을 기반으로 teacher model이 student model과 동일한 구성이고 장치에서 저장 및 실행될 수 있다고 가정함.
- 알고리즘 1
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/7dbb5568-ec47-4cac-8a19-059e0a4db0ab){: width="50%" height="50%"}{: .center}
    
    - 알고리즘 1은 self-learning method의 디테일을 보여주고 있음.
    - 각 training round에서 unlabelled audio를 device으로부터 들고옴.
        - 이때 teacher model을 통해서 label을 획득하는데 엄청 낮거나 높은 confidence을 들고있는 애들은 제외시킴
    - 여러번의 local update step들이 각각의 device에서 진행되거나 single gradient update가 시행될 수 있음.
    - gradient들은 unlabeled audio on-device을 통해서 획득되고, audio와 techer label의 augmented 된 버전임
    - server update step은 aggregaed local model delta를 수도-gradient으로 사용해서 update함
    - 최종적으로 각 training의 끝에서 teacher model은 EMA을 사용해서 update됨
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/26044245-85d8-4715-b63b-6f2de9f495d0){: width="50%" height="50%"}{: .center}
        
        - rehearsal training
            - model이 error feedback loop과 catastrophic 하게 older test set에 대해서 잊어버릴 수 있기 때문에, batches consisting of historical utterance with ground truth transcrition들은 unlabeled data를 이용하는 self-learning update에 사용될 수 있음.

## 2.3 Weak supervision

- weak supervision signal들은 시스템의 성능을 높이기 위해서 사용될 수 있음
    - 만약 사용자가 request를 멈추거나 취소하거나 반복한다면, 이거는 query가 device으로부터 성공적이지 않게 수행되었다는 것을 의미함.
    - 본 연구에서는 ASR model을 update할 때 이러한 feed back score을 반영해서 update하고자 했고, 이는 user’s의 요청이 성공적이었는지를 나타내는 잠재적인 요인일 수 있음
- 더 나아가서,  정확한 slot value에 속한 정확한  자연어 understanding semantics들이 회복될 수 있음.
    - user의 명시적인 re-invocation 으로 인함
    - 따라서 NLU slot label의 형태의 약한 feed back을 활용하는 것도 연구함
- 본 연구에서는 weak supervision label들에 대한 impact들을 2가지로 증명함
    1. NLU semantic으로부터 형성된 machine
        - alternate spoken language understanind built from ASR → NLU(natual language understanding) as a proxy for inferred semantics from user session data
            - ASR을 통해서 얻어진 transcrioption보다 더 weak한 정보를 사용함
    2. synthetic user feedback scores

### 2.3.1 Weak supervision: NLU semantics

- alternative AR과 NLU model으로부터의 generated 된 NLU semantic들은 weak NLU feedback으로 사용됨
    - 이전 선행 연구들은 rewriting utterance으로부터 생성된 NLU feedback을 사용했음
- NLU semantics z를 gt로 다루면서 우리는 semantic cost metrix M(z, y_i) 을 ASR hypothesis으로 계산할 수 있음.
    - semantic cost metric은 주어진 가설에 기반하여 계산되며 이는 오류가 존재하는 slot의 비율임.
        - 만약 slot속에 있는 token들이 hypothesis를 전부 대표하지 못한다면 해당 slot은 에러가 있는 것으로 간주함.
- 실험적으로 본 연구의 저자들은 alternate system의 ASR transcript 을 NLU semantic에 적용하고자ㅣ 했음.
    - 이러한 경우에 cost M은 WER을 포함할 수 있음.
- NLU semantic의 오류에 대한 정보를 feedback으로 주기 위한 모델 훈련을 진행함.
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/589a2c31-3427-4b37-b11a-567f0a2037ae){: width="30%" height="30%"}{: .center}
    
    - z : ground truth
    - y와 z에 대한 WER을 비교하는 것

### 2.3.2 Weak supervision : Feedback scores

- section 2.3.1에서 우리가 weak 한 NLU semantic을 얻을 수 있다고 가정을 하였음. 따라서 우리는 어느 hypothiesis y_i에 대한 feedback을 받을 수 있게 됨.
- 여기서 우리는 weak supervision이 user에게 제공된 가설에 대해서만 얻을 수 있다는 제약이 따라옴.
- 이러한 가설 아래에서 feedback score을 기반으로 한 weak superviison은 user feed back을 좀 더 자세히 가까이 simulate하고자 함.
- 본 연구에서는 2가지 형식의 피드백 점수를 사용함
    1. the semantic score as detailed in section 2.3.1 applied only to the served hypothesis
    2. binary feedback cost based on the sentence error rate with the true transcription
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/f9d62f39-aa68-4565-8793-6310ea1358a3){: width="30%" height="30%"}{: .center}
        
        - Binary feedback cost 는 $M(y,z_t) = 1(y!=Z_t)$ , reward function은 $M1(y,z) = M(y,z_t)+U$ 여기서 $Z_t는 실제, y는 예측$

# Experiments

- RNN-T 와 specaugment를 사용한 모델을 기반으로 함
- 실험 결과, federated self-learning with weak supervision 을 사용했을 때, WER이 낮아짐을 보임
- voice assistant data에서 update가 이루어질 경우 성능 improvement가 일어나지 않음
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/8c347028-4f73-4c96-bd79-e70d8e20a8bb){: width="30%" height="30%"}{: .center}
    

# Conclusion

- ASR을 위한 federated continual learning을 제안함
    - human이 주는 ground truth가 필요 없고, 디바이스의 자원이 좋지 않아도 사용가능함을 주장.
    - 또한 cloud storage을 사용하지않고 그냥 ondevice에서의 프로세스가 가능함을 장점으로 가져갈 수 있음
- EMA를 통해서 업데이트된 paried teacher model을 사용해서 self-learning을 함. → ASR의 성능 향상
    - 특히  unseen data에대한 성능이 많이 올랐음
- 근데.. ondevice을 위한 논문이라면서 ondevice에서 실험한 내용이 없음.



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