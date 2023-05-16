---
layout: post
categories: 논문리뷰
tags: Timeseries
comments: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] Clustered Hybrid Wind Power Prediction Model Based on ARMA, PSO-SVM, and Clustering Methods"
---


##### 논문 및 사진 출처
>Wang, Yurong, Dongchuan Wang, and Yi Tang. "Clustered hybrid wind power prediction model based on ARMA, PSO-SVM, and clustering methods." IEEE Access 8 (2020): 17071-17079.

<aside>
💡 기존에 있던 ARMA 모델과 PSO-SVM 모델을 합쳐서 시계열 예측 모델을 구성함. 
여러개의 모델을 합쳐서 전반적인 풍력 예측의 성능을 놓쳤고, 본 논문에서는 시계열 방법과 inteligent method를 합쳐서 좀 더 값진 정보를 얻음과 동시에 정확도를 향상시킬수 있었다고 함

</aside>
# 0 Abstract

- wind power prediction의 필요성 언급
- 본 연구에서는 과거 데이터(풍력과 기온)을 통해서 ARMA(autoregressive moving average) 예측 모델과 SVM prediction model을 설계함.
    - PSO(Particle Swarm Optimization) 알고리즘이 SVM 모델의 파라미터 optimization에 사용됨
- ARMA 에 기반한 PSO-SVM-ARMA의 하이브리드 예측 모델과 PSO-SVM 모델은 풍력 예측을 위해서 illustrated되어 있음
- 최적 가중치를 찾기 위해서 covariance minimization 방법과 PSO를 사용함
- 클러스터링 이론을 기반으로 시계열을 클러스터링함
- 풍력 예측을 위한 효과적인 dataset을 조사하고 clustered hybrid PSO-SVM-ARMA(C-PSO-SVM-ARMA) 풍력 예측 모델을 제안

# 1 Introduction

- wind power은 최근 10년간 비약적인 발전을 이루었음
- wind power의 stochastic 측성을 반영하기 위해서는 power system은 wind power 소비를 최대한 보장하면서 powergrid(전력망)의 안정적인 배차(dispatch) 와 안정적인 작동이 보장되어야함
    - 이를 해결하기 위해서는 전통적인 generator의 capacity의 자원을 늘리는 방향이 있음
    - 또다른 효과적인 방안은 wind power predicton의 정확도를 향상시키는 것
    - 첫번째방안의 장치 유지비용과 투자 비용이 너무나도 큼
- wind power prediction을 통해 power dispatch는 효과적이게 될 수 있음
- 시계열 분석은 두개의 주요 함수를 포함하고 있음
    - 예측 함수
    - description 함수
        - **다른 모델을 만듦으로서 research object이 어떻게 변화하는지 확인할 수 있음?**
- 전통적인 시계열 예측 방법은 continuous method, moving average method, autoregressiv emoving average method(ARMA) , GARCH method가 있음
- GARCH model은 시계열의 더 높은 특성을 묘사할 수 있고, 효과적인 ARMA 모델에 기반함
- SVM은 한정적인 정보를 기반으로 괜찮은 예측을 함
- 본 논문에서는 ARMA 와 SVM이 사용되었으며, 더 나은 hybrid wind power prediction model을 구축하기 위해서 prospectively combined 되었으며 clustered 되었음
    
    

# 2 The hybrid wind power prediction model base don ARMA and SVM

## 2.A The ARMA model

- ARMA 모델은 stationary time series에서부터 relevant 한 정보를 추출할 수 있음
- 제한적인 과거 데이터를 기반으로 효과적인 예측 결과를 얻을 수 있음
- xt = stationary time series이고 , ARMA(p,q) model은 아래 식과 같음
    ![Untitled](http://drive.google.com/uc?export=view&id=1zv87RccmmGFfwGWVw_wTcH8s9GLsy9o_){: width="80%" height="80%"}{: .center}
    
    - ai = autoregressive parameter
    - bj = moving average parameter
    - least square esitmation으로 평ㄱ되었음
    - et= 가우시안 손실
    - p = AR(auto regressive) 의 순서
    - q = MA(moving average) 의 순서
    - ARMA모델은 현재 예측 값을 과거 p 의 값들과 q의 과거 교란(disturbance)의 선형 결합으로 예측함
- ARMA 모델을 보다 정확하게 하기 위해서는p,q의 범위를 시계열 특성에 따라서 먼저 식별해야함
- 더 높은 order(p,q)를 설정하면 더 좋은 예측을 내놓지만, 더 많은 컴퓨터 자원을 사용하게 됨.
    
    ![Untitled](http://drive.google.com/uc?export=view&id=10v2v2qs_9DoqHW_VK6SToGlNrjMcMSZG){: width="80%" height="80%"}{: .center}
    
    - AIC, BIC라는 것을 통해서 ordering 범위를 설정해줌
    - 가장 작은 AIC와 BIC는 만족스로운 모델을 얻을 수 있다고 함
    - AIC 와BIC중 선택 방법
        - 시계열의 길이가 늘어남에따라 BIC의 복잡도는 더 높아짐. 따라서 BIC를 통해서 선택된 order의 범위는 좁을 것
        - prediction model의 정확도를 향상시키기 위해서, AIC가 더 합리적이라고 주장하고 본 논문에서도 AIC 가 사용됨
- ARMA를 모델링하는 과정에서 시계열은 주로 정적임. (1) 식에 기반하면 ARMA모델은 et가 가우시안 분포를 따를때만 설계될 수 있기 때문에 본 논문에서는 et가 white gaussian nosie라고 가정함

## 2.B The PSO-SVM Model

- wind speed, temerature, air pressure 등의 influence factor을 사용하기 위해서 본 논문에서 SVM이 사용되었음
- SVM의 기본적인 원칙은 sample set {(xi, yi), i = 1,2,3,4,..N)}, xi 는 R^n에 포함, 는 입력된 샘플 공간을 represent 하고, yi 는 R, N에 포함, yi는 샘플 데이터의 사이즈에 관련있다는 것
- 기본 아이디어는 인풋 데이터의 벡터를 고차원 피쳐 차원에 맵핑하고자 하는 것이고, 피쳐 공간의 선형회귀를 얻기 위해서 아래의 식을 사용함
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1fdd6Z6PRgn98hTxm9yx35Jv1Ac4VSM7H){: width="80%" height="80%"}{: .center}
    
- o(x)가 맵핑 함수, wt가 training dataset을 통해서 예측된 파라미터, b는 상수 coefficient.
- 피쳐 공간의 선형 회귀는 5,6의 constraints를 따라서 위험 최소화 규칙을 따름
    
    ![Untitled](http://drive.google.com/uc?export=view&id=11kSnFFQjKkP-kZonJ8qbRcPmh8jnqcy4){: width="80%" height="80%"}{: .center}
    
    - xi가 입력 벡터이고, i i*은 relaxation factor이며 이건 loss coefficient e와 관련이 있음.
    - C는 회귀의 질을 control 하기 위해서 사용된 패널티 팩터임
- 고차원 공간을 위해서 내적 함수인 o(xi)는 kernel functon K(x,xi)로 치환될 수 있고 본 논문에서는 가우시안 radial basis function(RBF) 커널 함수가 SVM 모델에 사용되었음

## 2.C The combination of ARMA and PSO-SVM Model

- 각각의 예측 모델은 다른 측면에서 장점을 보이고 다양한 예측 성능을 보여주기 때문에, single prediction모델을 독단적으로 평가하기는 힘듦. 그런데 두개나 세개의 효과적인 예측 모델을 합쳐서 전반적인 풍력 예측의 성능을 높이는 것은 좋은 시도임
- ARMA와  PRo-SVM 모델을 합친 것은 시계열 방법과 inteligent 방법을 합친 것으로 좀 더 값진 정보를 얻을수 있음과 동시에 정확도를 향상시킬 수 있음
- 아래 식으로 표현 가능
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1r0zs2GmaXHBzlyMwoDnZncJqV5vyP4qz){: width="80%" height="80%"}{: .center}
    
    - w1 + w2 = 1
    - Fcomb = 하이브리드 모델의 결과값
- 공분산 최소화 방법이 PSO-SVM-ARMA 예측 모델을 설정하기 위해서 사용되었으며, modified weight combination solution으로 동일 가중치 방법임
    - 9번 식이 최소화 되어서 가중치 coefficient들이 최적화될 수 있도록 세팅함
        
        ![Untitled](http://drive.google.com/uc?export=view&id=1s4bBWySs52LJE8jY9nyf6l5tr2TDkKXE){: width="80%" height="80%"}{: .center}
        

## 2.D The modeling process of PSO-SVM-ARMA Model

- PSO-SVM 모델이 먼저 학습되고 파라미터들이 설정/최적화됨
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1Ej9qtzXSUAAjHR3sFeIrQ7KoUfMd5ZJ3){: width="80%" height="80%"}{: .center}
    

# 3. Clustered hybrid wind power prediction model

- intermittent 하고 확률적인 특성을 가지고 있는 wind power으로 인해, 과거 데이터의 시계열을 탐구해보고 정확도를 높이기 위한 방안을 탐구하는 것은 좋은 시도임

## 3.A The clustered hybrid prediction model

- 논문에서 제안한 모델은 과거 데이터에 기반함
    
    ![Untitled](http://drive.google.com/uc?export=view&id=/1Ok11D69Laz4s2H2DdTOhDhJYnAhWp3Nw){: width="80%" height="80%"}{: .center}
    
- 클러스터링 이론에 따라서, 일년치의 과거 데이터는 열개의 클러스터링으로 나누어지게 됨
    - **이 때 각기 다른 시간과 특성에 따라서 나누어지게 되고, 본 논문에서 제안된 모델은 각기다른 파라미터를 통해 각 클러스터를 기반으로 학습됨**
        - 클러스터가 기준이 되어서 학습된다는 말인듯?
        
        ![Untitled](http://drive.google.com/uc?export=view&id=1ZLMjisU_v6urEDLD7DcUDwLkpTMWzWtm){: width="80%" height="80%"}{: .center}
        
    - 풍력 예측이 필요한 경우, 예측 데이터가 속한 클러스터를 판단하고, 과거 데이터는 예측을 위해서 해당 클러스터 모델에 입력됨
    - DTW
    
    wind 값은 intermittent 데이터여서 clustering을 통해서 이를 해결함. 아마 0인 값과 아닌 값이 clustering으로 구분됨.

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
