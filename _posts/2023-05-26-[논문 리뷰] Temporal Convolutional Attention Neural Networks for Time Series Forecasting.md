---
layout: post
categories: 논문리뷰
tags: IO, SensorData
comments: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] Temporal Convolutional Attention Neural Networks for Time Series Forecasting"
---


##### 논문 및 사진 출처
>Lin, Yang, Irena Koprinska, and Mashud Rana. "Temporal convolutional attention neural networks for time series forecasting." 2021 International joint conference on neural networks (IJCNN). IEEE, 2021.


<aside>
💡 태양열 관련 예측 모델 제안 연구

</aside>
# Abstract

- Temporal Convolutional Neural Networks(TCNNs)은 time series forecasting을 포함한 여러가지 sequence modelling task에 적용되어 왔음.
- 하지만, 인풋이 긴 경우, TCNN은 엄청 많은 ConvLayer을 요하며 interpretable result을 제공하지 못함.
- 본 연구에서는 solar power forecasting을 위한 TCAN 을 제안함
- TCAN
    - TCNN의 계층적 conv 구조를 사용해서 temporal dependencies를 추출하고, sparse attention을 사용해서 중요한 timestep들을 추출함
    - TCAN의 sparse attention layer는 extend된 recpetive field를 가능하게 함(deep한 구조 필요 x)

# Introduction

- ARIMA 와 같은 통계적인 방법은 well -established되었고 폭넓게 사용되었지만 모델을 선택할 때 도메인 조식을 요하고, 각 시계열을 독립적으로 fitting하며 관련 시계열에서 공유하는 패턴을 추론할 수 없다는 단점이 있음.
- 반면, 딥러닝 방법은 time series forecasting에 엄청나게 많이 적용되어 왔고 promising한 결과도 보여주었음. 딥러닝 기반 방법들은 rawdata를 less 도메인 지식인 상태로 학습할 수 있게하였고 복잡한 패턴을 추출할 수 있게 했음
- TCNN
    - sequence modeling task를 위해서 spcifically design 된 모델임
    - casual conv, dilated conv, residual connection을 사용해서 larger receptive filed를 구축하고 unstable한 gradient problem을 축소함과 동시에, 학습 속도의 향상을 이루게 함
    - 하지만 만약 input sequence가 길다면 TCNN은 temporal conv layer을 통해서 충분히 넓은 recpetive field를 구축해야할 것임.
    - 추가적으로 TNCN은 black-box 구조이기 때문에 해석할 수 있는 result를 주지는 않음
        - 근데 많은 time series forecasting은 critical한 결정들을 포함하고 있고, 결정권자에게 충분한 신뢰도를 주어야함.
- 본 논문에서는
    - Conv 구조와 attention 방식이 합쳐진 TCAN을 제안.
        - TCAN은 계층적 conv 구조를 통해서 temporal한 dependency를 학습하고, sparse attention layer을 통해서 forecasting result을 생성함
        - sparse attention layer을 사용함으로서 모델이 모든 historical input step에 접근가능하도록 하고, 가장 중요한 timestep에 집중할수 있도록 함과 동시에 결과를 visualization 해서 해석가능하게 함
    - TCAN을 세가지 현실 세계 태양열 데이터셋을 통해서 검증했고 sota를 찍음

## Case Study : solar power forecasting

- 태양열 예측은 GENERATOR의 최적화 스케줄링과 SOLAR을 ELECTRICITY GRID으로 통합하기 위해서 필요함

### Data

- 본 연구에서는 Sanyo, Hanergy, Solar데이터를 사용함
- Sanyo와 Hanergy는 호주에 있는 2개의 PV plant으로부터 발생된 데이터임
    - 오전 7시 ~ 오후 5시 사이의 데이터만 사용되었으며, 30분 간격으로 모아졌음
    - 두 데이터셋 모두 날씨와 기상 예보 데이터도 모두 수집되었으며 covariate으로 사용되었음
- Solar 데이터는 알바마와 미국에 있는 137 PV pant에서 수집된 데이터로, 1시간 간격으로 모아졌음
- 모든 데이터셋은 평균 0, unit 분산을 가지도록 정규화 되었음

### Problem statement

![Untitled](http://drive.google.com/uc?export=view&id=1mTqf1b9h39isQQmrQWIAXcdPEmuQBsKL){: width="80%" height="80%"}{: .center}

![Untitled](http://drive.google.com/uc?export=view&id=1_Szri1iE6606yaxisRC8bVGkZImWW6o4){: width="80%" height="80%"}{: .center}

- step t에서의 input은 yi, t-1, xit의 concat임
    - yi : i번째 PV power generated at time t
    - xit: time-based 멀티차원의 covariate vector

# Background

## Temporal Convolutional neural Network

- TCNN은 3가지의 main tech를 사용함; causal conv, dilated conv, residual connection
    - Causal conv
        - t에서의 output은 이전 layer의 earlier time steps 혹은 time t을 사용해서만 분해됨
        - zero padding이 hidden layer에 사용되어서 hidden layer가 input layer와 같은 차원을 가져서 convolution이 용이할 수 있도록 함
    - Dilated Conv
        - 넓은 receptive field를 가능하게 하였고 결과적으로 롱텀 메모리를 포착할 수 있게했음
        - sequence element s 에서의 dilated conv operator F는 다음과 같이 정의됨
            
            ![Untitled](http://drive.google.com/uc?export=view&id=1DSQONXAgJ3SrrnIX0BFG_IOd82-0eRoa){: width="80%" height="80%"}{: .center}
            
            - f는 conv filter
            - x는 sequential 한 input
            - k는 filter size
            - d 는 dilation factor
            - Con kernel은 모든 레이어에대해서 같지만, dilation factor은 네트워크의 깊이에 따라서 기하급수적으로 증가함.
    - Residual connection
        - Residual block은 gradient vanishing problem을 해결할 수 있도록 도움. 주된 아이디어는 x를 stacked layers에 input해준다는 것.
            
            ![Untitled](http://drive.google.com/uc?export=view&id=1yxkJDieUspVztdFt3FvIE66M_fvgLnB3){: width="80%" height="80%"}{: .center}
            
            ![Untitled](http://drive.google.com/uc?export=view&id=124a_3I4vcP82JspMENV4wntn6e48tyrn){: width="80%" height="80%"}{: .center}
            
            - 좌/ 우 두가지의 브랜치가 있는데, 두가지의 width가 같지 않기 때문에, 1*1 Cnv를 사용해서 우측 브랜치의 width를 조정해줌
    - dilated conv을 이용하면 더 넓은 receptive field를 가질 수 있지만, long input sequence를 효율적으로 다루지는 못함(더 많은 레이어, 더 많은 학습시간, 더 복잡한 구조)

## Attention Mechanism

- attention mechanism은 원래 seq2seq 테스크를 통해서 처음 제안되었음
- 이는 encoder-decoder 프레임워크에 사용되어서, 자동으로 encoder input 시퀀스에서 decoder output에 많은 영향을 주는 부분을 identify할 수 있도록 함
- seq2seq framework에서는 encoder와 decoder가 sequential한 step을 가지고 매 시점마다 hidden state을 생성하도록 함.
- soft attention은 encoder의 hidden state과 decoder의 hidden state을 input으로 받고, context vector을 생산해냄
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1ajt7HBfoyZyydHAFENRYqNkvIhrkJzqG){: width="80%" height="80%"}{: .center}
    
- 이후 softmax함수를 통해서 정규화되고 attention 가중치를 생성하게 되며, 매 encoder의 hidden state hi에 대한 weight ai의 식은 아래와 같음
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1yu5EA8JOyNuEa1n0nyqKF0h2LZyE2wkU){: width="80%" height="80%"}{: .center}
    
    - 이 가중치는 encoder step i 가 decoder output setp t에 대해서 가 얼마나 중요한지를 나타냄.
- 최종적으로 attention layer의 output은 attention wieght 과 encoder hidden state을 dot product해서 얻어지게되며, weighted output은 decoder의 hidden state과 concat되어서 decoder의 output을 생성하게 됨
- 직관적으로 보면, attention mechanism은 decoder가 historical sequence step에서 가장 중요한 부분에 focus할 수 있도록 도와주고, seq2seq의 한계점을 극복할 수 있도록 했음
- 하지만, softmax를 이용한 attention은 항상 positive한 attention weight을 모든 timestep마다 내놓는데, 이는 long sequence에 사용할 때 무관한 step도 포함될 수 있도록 함. 이를 극복하기 위해서 최근의 연구들은 sparse attention mechanism 을 제안해서 sparse한 attention mapping을 배울 수 있도록 했음.

# Temporal Convolutional Attention Neural Networks

## Motivation and Novelty

- TCAN aims to:
    - conv layer의 개수를 늘리지 않고도 large 한 receptive field를 가능하게함
    - input sequence의 time step 중 중요한 정보에 집중하고 불필요한 정보는 무시함
    - 가장 관련있는 timestep에 대하여 시각화를 제공함
- TCNN이 recpetive field를 넓히기 위해서 expoentially dilated conv 를 사용함에도 불구하고, 만약 인풋 시퀀스가 길다면, conv layer가 필요하게되고, 이는 학습 시간 증가와 복잡성을 높이게 됨
- TCNN이 예측하기 위해서 필요한 input step의 개수는 Conv에 있는 모든 effective 한 history들을 합친 개수임
- Tl길이만큼의 input sequence를 커버하기 위한 receptive fild를 위해서는, TCNN은 최소한 nl 개의 conv layer가 필요함
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1EKdh-s9eq9Ohs2-wb9Bj73gpnAh63_Yz){: width="80%" height="80%"}{: .center}
    
    - k = 커널 사이즈, dl = dilation factor, Tl : input sequence length, nl : conv layer 개수
    - 본 연구에서는 tl이 20~24, k  = 3이기 때문에, 4개의 conv layer가 필요함

## Model Architecture

![Untitled](http://drive.google.com/uc?export=view&id=1DZCa0TqemtaFKGXgulwF5nUXOFaAXHBb){: width="80%" height="80%"}{: .center}

- TCAN은 그림과 같이 3개의 부분으로 나누어져 있음
    - temporal conv layer
    - sparse attention layer
    - output layer
- TCAN은 계층적 conv 구조를 사용해서  input sequence를 encode하고, temporal pattern을 latent variable으로 추출함
    - latent variable은 attention mechanism에서 가장 relevant한 feature을 학습하고 final prediction을 생성하는데에 사용됨
    - latent variable은 전체 input window의 정보를 encode하고, 추가적인 conv없이도 넓은 receptive field를 가지도록 함

### Temporal Conv Layers

- TCAN은 temporal latent factor (ht-Tl:t)를 multiple dilated temporal conv layer(TC)들을 통해서 추출함
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1V8KDvNr7i9fN8XTwv9hnaur4ckVlo6o8){: width="80%" height="80%"}{: .center}
    
- 추출된 latent factor은 intput sequence의 모든 정보를 encode함

### Sparse Attention Layer

- temporal latent factors(ht-tl:t)를 input으로 받고, predcition을 위한 attention vector(ht)를 생성하도록 함
- Transformer나 RNN구조에 사용되었던 전통적인 attention score은 softmax 함수로 계산되었음. 하지만, softmax는 previous timestp들에 대해서 0을 절대 부여하지 않기 때문에, 중요하지 않은 부분들을 완전하게 배제하지는 않음
- 시퀀스 모델링 테스크에서는 future timestep가 몇몇개의 이전 timestep에 강하게 관련되어 있음
    - ex. t시점에서의 solar power는 같은 날 몇시간 전 시점이나, 다른 날 같은 시점의 solar power과 강하게 연관되어 있음
        
        ![Untitled](http://drive.google.com/uc?export=view&id=1x72alOIqD4ptxzt-4D5sSA_WWMbouMQu){: width="80%" height="80%"}{: .center}
        
- 본 연구에서는 알파-entmax attention을 적용했고, 아래 식과같이 정의됨
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1KRsRY55jCTyn_AemA6-y4GOlsro7E9kS){: width="80%" height="80%"}{: .center}
    
    - r : 큰곱하기
    - 1 : all-one vector
    - 알파 : 하이퍼파라미터
    - 알파-entmax 는 알파가 1이고 softmax일 때랑 알파가 2이고 sparsemax를 사용할때 동일함
        
        ![Untitled](http://drive.google.com/uc?export=view&id=1iDsDpyHNCY2PE5WGMWK8EOVtKVugXJPn){: width="80%" height="80%"}{: .center}
        
        - ct는 attention score과 hidden state의 dot product한 결과값

### Output layer

- output layer에서는 attention vector들을 사용해서 final prediction을 만듦
- 본 연구에서는 데이터가 가우시안 분포를 따른다고 가정하는데, 가우시안 분포는 real-world time series 모델링 시 자주 사용됨
- attention vector을 분포의 평균과 variance를 포함하는 예측 결과로 transfer함
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1D8Zgj9rw8Dw2b35wuB28w0qb8SZDYdwv){: width="80%" height="80%"}{: .center}
    
    - 11번 식은 variance가 항상 양수가 되도록 만들어줌
    - 10, 11번 식을 통해서 가우시안 분포를 형성하고, 분포에서 prediction이 샘플링 될 수 있게됨

### Loss Function

- 아래 식을 통해서 loss 가 최소화됨
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1baGEc7KUEy5sVpZQ_Oal7WoOTNp7FwDo){: width="80%" height="80%"}{: .center}
    
    - y^는 point 예측
    - point와 probabilistic 예측을 모두 다 정확하기 위해서 MAE 와 Negative Log-Likelihood(NLL)을 합쳐서 사용함.
    - 정규화 파라미터 a를 사용하였고, a가 커질수록 probabilistic forecast의 가중치가 높아짐

# Experimental Setup

## Methods used for comparison

- TCGAN 을 Deep AR, N-Beats-G, N-Beats-I, LogSparse Transformer, TCNN과 비교함
    - Deep AR
        - 광범위하게 이용되는 seq2seq probabilistic 예측 모델
    - N-BEATS
        - backward 와 forward residual link들과 fully connected layer들의 stack들에 기반한 모델
        - N-BEATS-G는 generic forecasting 결과를 , N-BEATS-I는 interpretable 한 결과를 보여줌
    - LongSparse Transformer
        - 최근에 제안된 transformer의 time series forecasting에 맞도록 변형된 모델
    - TCNN
        - 새로운 conv 구조
    - Persistence
        - typical baseline in forecasting
        - next day의 prediction을 위해서 이전 날짜의 time series 를 고려함

### Data Split and Hyperparameter Tuning

- 아담 옵티마이저를 사용했고, mini batch gradient descent를 사용해서 최적화되었으며 에폭은 200으로 설정됨
- 베이지안 optimization을 하이퍼파라미터 search로 설정하였으며, 최대 iteration은 20으로 설정함
- 비교대상에 있는 모델들은 원본 논문에 제안된 방법대로 파인튜닝됨
- Sanyo와Hanergy데이터셋에 대해서, 마지막 년도에 해당하는 데이터는 test , 두번째 마지막년도에 해당하는 데이터는 validation set, 남은 데이터들은 training에 사용함
- TCAN에서 lr은 0.005로 고정되어 있었고 배치사이즈는 Sanyo에서는 256, Hanergy와 Solar에서는 512로설정됨. 알파는 1.5, 정규화 파라미터는 0.5로 설정됨. dropoutrate은 0,0.1,0.2중하나로 선택되었고 커널사이즈는 3,4중 하나로 선택되었음.

### Evaluation Measures

- Accuracy results
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1cGsTGa4iVyOKeA0cTICEW4iTBi_loaIt){: width="80%" height="80%"}{: .center}
    
    - Persistence와 N-Beats가 probabilistic forecastfmf todtjdgkwl dksgrl Eoansdp, 0.5 loss(==MAE loss)만 결과가 나옴
    - Point forecast에서는
- 두개의 TCNN 모델을 비교했을때, TCNN-4가 TCNN-3보다 우월한 것을 알 수 있음
    - TCAN은 두개의 모델보다 더 정확하고, 위에서 언급했던 정확도 향상 및 receptive filed의 확장을 가능하게 했음(Conv추가 없이)
    - TCAN과 TCNN-4모두 인풋 시퀀스를 커버할 수 있지만, TCAN이 더 적은 Conv를 사용함
- 2개의 consecuative days의 예측 결과
- consecuative
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1pDTFPPK_PVEDDzV6P7s1TvEVk33kYEJL){: width="80%" height="80%"}{: .center}
    
    - 좌측은 actual vs predicted values for each day
        - 얼마나 잘 예측했는지를 보여줌
    - 우측은 corresponding sparse attention map
        - 이건 pair attention score을 보여주는데, 이전 time series step중에 미래를 예측하는데에 중요도를 나타냄
        - 과거와 미래 step에 대한 의존성이 sparse하다는 것을 보여주며, 더 긴 과거에 access할 수 있다는것이 중요ㅏ다는 것을 보여주었음
            - EX. 모든 MAP이 TIME STEPS의 이른 시점에서는 엄청 높은 ATTENTION SCORE을 보여줌
                - 첫번째 future prediction은 second input step으로만 결정됨
    - TCAN과 TCNN-4의 학습 속도를 비교했을 때, TCAN이 상대적으로 빠른 학습을 보여주었음
        
        ![Untitled](http://drive.google.com/uc?export=view&id=14_bLE3gLjecVXN636nLWsyFfk9KcpvQl){: width="80%" height="80%"}{: .center}
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