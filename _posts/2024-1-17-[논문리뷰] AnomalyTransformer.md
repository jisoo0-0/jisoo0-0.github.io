---
layout: post
categories: 논문리뷰
tags: anomaly_detection, time-series, transformer
comments: true
math: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] Anomaly transformer: Time series anomaly detection with association discrepancy"
---
출처
> Xu, Jiehui, et al. "Anomaly transformer: Time series anomaly detection with association discrepancy." *arXiv preprint arXiv:2110.02642* (2021).

Copyright of figures and other materials in the paper belongs to original authors.

# 0 Abstract

- 시계열 task에서의 anomaly point를 비지도 학습 방법으로 탐지하는 것은 challenging함
- transformer 는 pointwise representation이나 pairwise association 을 unfied modeling하는데에 좋은 성능을 보였고, 연구진들은 각시점에서의  self attention weight 분포를 사용해서 rich association을 생성해 낼 수 있다고 주장함
- 본 연구의 Key observation은 anomaly가 희소하다는 점에 의거한 것으로, abnormal point들의 사소한 association들을 building하는 것이 어렵기에 인접한 시점들에 집중해야한다는 것임
    - anomaly자체가 조금 희박해서 해당 point들의 특징같은 것을 잡아내서 모델링하기 힘드니까 주변 시점들을 자세히 보고 그걸 비교해서 찾아내겠다는 말같음
- 본 연구에서는 anomaly transfoermer 을 새로운 anomaly attention mechanism을 사용해서 제안함
    - association discrepancy를 계산하기 위한 방안임

# 1 Introduction

- 우리의 현실세계는 항상 연속적이게 흘러감.
    - multi sensor을 통해서 연속적인 데이터들을 축적가능
- large scale system의 모니터링 데이터를 통해서 오작동을 검출해 내는 것은 security를 ensure할 수 있고 재정적인 loss를 피하게 해줌
- OC-SVM, LOF, SVDD와 같은 anomaly detection의 classic한 방법들은 temporal한 정보들을 고려하지 않고 실제 시나리오의 실제 데에터에 일반화되기 어렵다는 문제점들이 있음
- NN의 capability에 따른 representation을 learning하는 것은 superior performance를 보여주었지만, 대부분의 방법들은 잘 고안된 recurrent netowrk을 통하여 pointwise representation을 학습하는 방법이었음
    - + reconstruction을 통한 self supervised 방법이거나 auto regressvie task였음
    - 따라서 anomaly criterion이 pointwise reconstuction이나 prediction error였음
    - 하지만 anomaly 데이터가 많이 없기 때문에 pointwise representation은 복잡한 시계열적인 패턴을 잡아내기에는 부족하고, 아무래도 데이터 양이 많은 정상적인 time point들에 의해서 dominated 될 수 있음
    - 또한 reconstruction이나 prediction error은 point별로 계산이 되는데, 이건 temporal 한 context를 종합적으로 반영해줄 수 있는 방안이 아님
- 명백한 association modeling을 통해서 anomaly를 detect하는 방법도 잇음
    - vector autoregression과 state space model이 이 범주에 속함

- 그래프도 temporal한 pattern을 capture하기 위한 방법으로는 부족했는데, 이는 learned graph이 single time point에 limited되었기 때문임
- 본 연구에서는 Transformer  을 tiem series anomaly detection에 적용함
    - Transformer을 time series에 적용하면서 연구진들은 **self attention map**을 통해서 각각의 time point의 **temporal association**을 얻을 수 있음을 발견함.
        - **temporal association의 분포**는 temporal context에 대한 유용한 description을 제공해 줄 수 있음
        - **period or trend of time series 같은 정보 제공**
        - 해당 분포를 본 연구에서는 **series association**이라고 명칭함
- 아래의 observation에 기반해서 **association distribution의 normal - abnormal inherent**을 사용하고자 함
    - 연속성 때문에 인접한 time point에 **비슷한 abnormal pattern을** 보이기 때문에 이에 집중하려고 함
        - 이러한 연속적인 concentration inductive bias를 **prior-association**라고 부름
    - 반면에 dominating normal time point들은 전체 series 을 기반으로 informative association을 discover할 수 있음(인접한 영역에 한정되지 않음)
- 이와 같은 시도는 각 time point에 새로운 anomaly criterion을 적용하게 하였음
    - 새로운 anomaly criterion은 각각의 time point의 **prior association**과 각 time point의 **series association**간의 거리로 정량화 되는데 이를 **association** discrepancy라고 명칭함.
    - 이전에 언급된 것처럼, anomaly간의 assocaitation이 adjacent-concentrating 하기 때문에 anomaly들은 smaller association discrepancy를 보일 것
        - anomaly들이 서로 인접하게 모여 있을 것이기 때문에 prior association과 temporal 적인 context가 반영된 series association간의 거리가 짧을 것이라고 예상하는 것
- 본 연구에서 association discrepancy를 계산하기 위해서 self attention mehcanism을 anomaly attention으로 개조함
    - prior association과 series association각각을 time point에 대해서 모델링하는 것을 포함
    - prior association은 학습가능한 가우시안 커널을 사용해서 인접한 concentration inductive bias를 각 time point에 대해서 present하려고 함
    - series association은 raw series으로부터 학습된 self attention weight과 correspond됨
- 본 연구에서는 두개의 branch(prior association, series association) 에 minimax 전략이 적용됨
    - 이는 normal-abnoraml distinguishability를 amplify 해주고 새로운 association based criterion을 제안해줌

# 2 Related Works

- pass

# 3 Method

- 연속적으로 d번 측정하는 시스템을 모니터링하고 각 시간에 따라 동일한 간격의 observation을 기록한다고 가정함.
- 미리 언급했듯이 연구진들은 unsupervised time series anomaly detection의 핵심을 informative한 representation을 학습하는 것과 구분가능한 criterion을 찾는 것이라고 강조함

## 3.1 Anomaly Transformer

- Overall 구조
    - anomally transformer 은 anomaly attention block과 FFL을 반복하면서 쌓는 것으로 구성됨
    - 이러한 stacking구조는 deep한 multi level feature으로부터 underlying한 association들을 찾는데에 도움이 됨
    - model이 L개의 layer을 N개의 길이로 가지고 있다고 가정하면, l번째 layer의 방정식은 아래와 같음
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/2250d30f-83f2-4e5f-aa36-e25715c3d222){: width="50%" height="50%"}{: .center}
        
        - X는 input time series
        - Xl은 l번째 layer의 ouput
        - initial input X0은 raw series를 임베딩한 것으로 표현됨
            - X0=Embedding(x)
        - Zl은 l번째 layer의 hidden representation임
        - 식을 해석해보면 l번째 layer의 hidden representation은 이전 layer의 output에 anomaly attention을 적용하고, output과 더해서 layer normalization을 적용해 준 것이고
            
            l번째 layer의 output은, hidden representation을 FFL에 입력하고 출력값과 Zl을 더해서 layer normalization을 시켜준 것 
            
- Anomaly Attention
    - single branch가 prior association과 series association을 동시에 modeling할 수 없음
    - 따라서 아래 figure1과 같이 two branch sturcture 을 제안함
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/7ba69339-0142-496e-bf5a-c7d03aec9c30){: width="70%" height="70%"}{: .center}
        
        - **Prior association**
            - learnable **Gaussian kernel을 기반**으로 **relative한 temporal distance 에 대한 prior** 을 계산함
            - 가우시안 커널이 unimodal 적인 특징이 있기 대문에 이러한 구성은 인접한 horizon에 대해서 좀 더 attention을 반영할 수 있음
            - 또한 learnable scale 알파를 사용해서 prior association이 various time series pattern을 adapt 할 수 있도록 함 (가변적인 anomaly segment를 adapting하는 것이 가능)
        - **Sereis association**
            - raw series으로부터의 **association을 학습**함
    - 두개의 branch으로부터 각각의 time point에 대한 temporal dependency를 유지하게 해서 point wise representation보다 좀 더 유용한 정보를 제공해주고자 했음
    - l 번째 layer에서의 anomaly attention은 아래와 같이 계산됨
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/64aa4598-16dd-4641-8fd2-dca1d7b467b5){: width="50%" height="50%"}{: .center}
        
        - Initialization은 Q,K,V와 각각의 weight 설정부분
        - Prior Association은 learned scale 알파에 의해서 생성되고, i번째 time point에 대해서 j번째 point와의 association weight는 가우시안 커널을 통해서 계산됨
        - 위의 식에서도 확인가능하듯이 본 논문에서 말하는 Series association은 attention 을 적용한 weight 라고 볼 수 있음
- Association Discrepancy
    - prior 과 series association의 대칭화된 KL divergence가 association discrepancy임
        - 이는 두개의 distribution에 관한 information gain을 나타냄
        - mulitple layer에 있는 association discrepancy를 평균내어 multi level feature과 combine 해서 좀 더 유용한 feature을 만듦
            
            ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/3f720bbb-2318-4791-b743-09c2ce4f9dbd){: width="50%" height="50%"}{: .center}
            
            - P  : prior association
            - S : series association

## Minimax association learning

- 비지도 task에서는 model의 optimizing을 위해서 reconstruction loss를 사용함
- reconstruction loss는 series association이 가장 중요한 association을 찾을 수 있도록 도와줌
- noraml과 abnormal time point간의 difference를 증폭시키기 위해서 additional loss를 사용해서 association discrepancy를 증가시켜 주었음
- prior association의 unimodal property 때문에 discrepancy loss가 series association이 non adjacent data에 집중할 수 있도록 도와줌
    - 이건 anomaly의 reconstruction을 어렵게 만들고 anomaly를 좀 더 identifiable하게 만들어줌.
- 사용된 loss function은 아래와 같음
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/653282b6-f4e5-49da-861e-464e9c6758be){: width="60%" height="60%"}{: .center}

    
    
    
- Minimax strategy
    - association discrepancy를 직접적으로 maximize시키면, 가우시안 커널의 파라미터 스케일을 엄청나게 감소시킬 거고 prior association을 의미없게 만들것임
    - association learning을 보다 효과적으로 control하기 위해서 본 연구에서는 minimax strategy를 사용함.
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/14edc1f7-54c5-4fe1-8192-8e30f783dae5){: width="70%" height="70%"}{: .center}
    
        - minimize 구간에서는 prior association을 통해서 series association을 approximate하고자 함
            - 해당 구간에서는 prior association이 다양한 temporal pattern에 adapt될 수 있도록 함
        - maximize 구간에서는 series association을 통해서 association discrepancy를 증가하고자 함
            - 해당 구간에서는  seires association이 인접하지 않은 horizon에 대해서 좀 더 집중할 수 있도록 함
- Attention based anomaly criterion
    - reconsturction criterion에 association discrepancy를 normalize 해서 temporal represenation과 distinguishable association discrepancy를 모두 다 사용할 수 있도록 함
    - 최종적인 anomaly score은 아래와 같음
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/52d8f1fa-0901-44a9-a492-f1433959683d){: width="60%" height="60%"}{: .center}
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