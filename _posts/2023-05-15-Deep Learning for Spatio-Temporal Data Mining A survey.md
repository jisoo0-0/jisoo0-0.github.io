---
layout: post
categories: Graph
tags: SpatioTemporal
comments: true
---


#### 관련 프로젝트
Covid GCN    
##### 논문 및 사진 출처
>Wang, Senzhang, Jiannong Cao, and Philip Yu. "Deep learning for spatio-temporal data mining: A survey." IEEE transactions on knowledge and data engineering (2020).    

##### 사담
Covid 관련 GCN project을 위해서 reseach하는 도중, spatio-temporal data에 대한 deep learning 연구를 잘 정리한 survey를 찾아서 블로그에 기록한다. 


# 0 Abstract

- GPS, 모바일 기계, remote sensing 등의 기술 진볼 인해서 STD(Spatio-Temporal Data)가 많이 증가했음.
- STD를 통한 Data mining은 실생활 application(인간 mobility 이해, 스마트 이동수단, urban planning, 공공 안전, health care, 환경 관리 등)에 중요한 역할을 함.

# 1 Introduction

- STDM’s Application은 아래와 같은 카테고리로 나누어 질 수 있다. 
    - Environment and Climate
        - wind prediction
        - precipitation forecatsting(강우 예보)
    - Public Safty
        - crime prediction
    - Human mobility
        - human trajectory pattern mining
- Classical Data Mining을 STD에 적용했을 때, 성능이 엄청 안좋게 나오는 이유에 대해서 논문은 이렇게 설명한다. 
    - STD는 연속적인 공간에 임베딩되지만, 전통적인 데이터셋은 아니기 때문
    - STD의 패턴은 시공간의 특징을 반영하는데, 이건 전통적인 방법이 capture하기 힘든 정도의 복잡성과 데이터 상관관계를 포함하고 있음
    - 전통적인 통계 기반 데이터 마이닝에서는 데이터가 독립적으로 생성됨을 가정하지만, STD는 self correlated되어 있음
- STDM을 위한 전통적인 방안의 문제점은 feature engineering에 있다고 한다. 
    - 전통적인 방안은 그들 스스로 raw한 데이터를 처리하는데에 문제가 있기 때문이다. 
- 전통적인 방안을 사용하지 않고, Deep Learning을 통해 STDM을 하면 좋은점들
    - Automatic Feature representation learning
        - Raw한 데이터에서 feature을 얻을 수 있음
    - Powerful function approximation ability
        - Deep Learning 방법을 통해서는, any curves 나 복잡한 비선형 함수를 해결 할 수 있음(as long as deep learning 모델이 충분한 layer랑 neuron들이 있다면)
        - 딥러닝 방법은 복잡한 STD가 포함된 STD 마이닝 task를 해결할 수 있는 아주 복잡한 함수를 approximated 할 수 있음
    - Performing better with big data
        - SVM이나 결정트리는 smaller dataset에 더 적합함
- 2016 이후부터는 많은 연구자들이 STD 마이닝을 deep learning 모델을 통해서 해결하고자 했음

# 2 Categorization of Spatio-Temporal Data

## 2-1 Spatio-Temporal Data Types

- STD를 event, trajectory, point reference, raster, video 로 카테고리화 할 수 있다. 

### 2-1-1 Event Data (Ex. crimes, traffic accicents .. )

- discrete events로 구성된 데이터이다. 
- Event 는 (e, l, t)로 일반화 될 수 있다. 
    - e : event의 종류
    - l : event 가 일어난 장소
    - t : event가 일어난 시간
        
        ![Untitled](http://drive.google.com/uc?export=view&id=1Xf8Yn_ue4r50EN6t22MgGTUH-ANKgA1x){: width="80%" height="80%"}{: .center}
- Event Data의 응용 분야
    - criminology
        - urban crimes
    - epidemiology
        - **disease outbreak(질병 발생)**
    - transportation
        - traffic accidents
    - social network
        - social event and trending topics

### 2-1-2 Trajectory Data

- object가 시간에 따라 이동한 경로를 trajectory라고 한다. 
    - 주로 GPS와 같이 움직이는 object에 붙은 location sensor를 통해서 얻을 수 있음
    - (location, time)간의 연속적인 집합으로 표현 가능
        
        ![Untitled](http://drive.google.com/uc?export=view&id=1p7oz-tC-cLYy7z4_A-nPDXN-j0RcO6mt){: width="80%" height="80%"}{: .center}
        

### 2-1-3 Point Reference Data

- moving reference points의 집합을 통해서 생성된다.
    - 점이 이동하면서 데이터가 변하는 것을 말하는 것 같음
- EX : 기상 측정 풍선이 이동하면서 생성하는 기상 데이터나, 온도 센서에 의해 측정된 해수면 온도의 점 기준 데이터
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1fP-V9eNou2q6jwo8OWERiVqmlWiu4UrR){: width="80%" height="80%"}{: .center}
    
    - 두개의 타임스템프에 관한 figure

### 2-1-4 Raster Data

- 고정된 location에서 regular/irregular time point에 모아진 것이다.
- point reference data와의 근본적인 차이점은, point reference data 에서 location 측정 기준은 **시간별로** 계속 바뀌는데, raster에서는 바뀌지 않는 다는 점이다. 
    - EX: 도시별로 수집된 미세먼지 농도 데이터 , road 마다 수집된 traffic flow 데이터
    - 예시 사진
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1BxTGtziGf-RQLPIF0ZP5jwT0J48107QK){: width="80%" height="80%"}{: .center}
    

### 2-1-5 Video Data

- neigbor pixel들은 주로 비슷한 RGB 색상을 가지고 있기 때문에 높은 spatial correlation을 가진다.
- 연속적인 frame들은 주로 smoothly changed되기 때문에 높은 temporal dependcy를 가진다.
- 비디오는 주로 3차원 텐서로 구성될 수 있다. (시간, 가로, 세로)
- 비디오는 특별한 raster data로도 생각할 수 있다. 
    - 각각의 pixel에 센서가 있다고 생각하면(고정된 센서) , RGB 색상값은 연속적으로 센서로부터 획득됨
- 본 논문에서 자세히 다루지는 않는다고 한다. (CV 쪽이라서 ..)

## 2-2 Data Instances and Formats

- 같은 종류의 STD라도, 다른 종류의 data instances가 확보될 수 있다.

### 2-2-1 Data Instances

- 예시
    - event, trajectory, point reference 가 모두 event으로 예시화 될 수 있다.
        - trajectory를 여러개의 discrete points으로 나눌 수 있음
        - 시계열로도 instantiated될 수 있음
      
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1w2EYRXAGT4tNS0y96HeMdLvYGDzdY3J2){: width="80%" height="80%"}{: .center}
    
    - **Covid와 같은 경우, 현재 이미지로 변환되어 입력값으로 들어갈 예정이니까 Matrix(2D), Event으로 설명되는게 맞음**

### 2-2-2 Data Formats

- Data Format은 주로 딥러닝 모델의 입력값으로 들어간다.
- 딥러닝 모델은 다른 포멧의 input을 요하기 때문에, STD instance를 어떻게 represent할 것인지는 Data mining task과 deep learning model에 따라서 달라질 수 있다. 
- event는 본연적으로 point으로 표현될 수 있고, trajectory와 시계열은 모두 sequence로 표현될 수 있다.
- trajectory는 matrix으로도 표현 가능하다. 
- Spatial maps는 graph과 matrix로 모두 표현가능하다. 
    - 도시 지역의 traffic flow prediction을 할 때, 도시 지역의 transportation network 의 traffic data는 traffic flow graph으로도 표현 가능하지만, 지역 레벨의 traffic flow matrix 으로도 표현 가능함
- **2D matrix는 3D tensor 보다 simple 한 데이터 format이지만, spatial information을 잃는다는 단점이 있으며, 둘 다 raster data를 표현하기 위해서 많이 사용됨**

# 3 Deep Learning Models for Different types of ST Data

## 3-1 Preliminary of Deep Learning Models

- RBM, CNN, Graph CNN, RNN, LSTM, AE/SAE, seq2seq을 간단히 설명한다. 
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1SfUD6EKDH0OPcycsQSGKAb698m1FioKh){: width="80%" height="80%"}{: .center}
    

### [3-1-1 Restricted Boltzmann Machines (RBM)](https://angeloyeo.github.io/2020/10/02/RBM.html)

- RBM은 2개의 layer으로 구성된 stochastic(확률적) neural network이다. 
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1jtHXQGBJm2oxSD0m_i4cRrx5myoeAfYs){: width="80%" height="80%"}{: .center}
    
    - **무방향 이분 그래프**
    - RBM에 속한 모든 노드들은 가중치가 있는 엣지들로 연결되어 있음
    - 같은 계층에 있는 노드들은 연결되지 않음. (visible→visible, hidden→hidden을 말하는듯)
    - RBM은 input의 representation이나 binary code를 학습하고자 함
    - superevised, unsupervised 모두 가능

### 3-1-2 Convolutional Neural Network(CNN)

- CNN 구성 요소 : 인풋 레이어, CNN 레이어, 풀링 레이어, fully connected layer, 출력 레이어
- pooling layer이후에 정규화 레이어 추가 가능
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1Mw84rBIRNTaa9uxYTZbyHDqsMqI5dmiN){: width="80%" height="80%"}{: .center}
    
    - raw image 를 input으로 받으면, CNN layer가 high level latent feature을 여러개의 kernel(filter)을 통해서 학습하게 됨
    - 이후, high level feature들이 pooling layer으로 input되고, spatial 차원을 통해 downsampling 작업이 진행되며(차원축소 말하는듯), 이 때 파라미터의 수가 줄어든다
    - 마지막으로 여러개의 fully connected layer들을 통해서 input latent feature들에 대한 비선형 transformation을 진행함
    - **Due to the powerful ability in capturing the correlations in the spatial domain, CNN is now widely used in learning from ST data, especially the data types of spatial maps and ST rasters.**

### 3-1-3 Graph CNN(GCNN)

- Pooling operation에 따라서 각 노드의 이웃 노드들에 대해서 convolutional transformation을 적용할 수 있다. 
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1Vu5z1RrVGfqLlHiVAeNDupRLtzPt0qQ1){: width="80%" height="80%"}{: .center}
    
    - 여러개의 graph convolution layer을 stack 함으로서, 각 노드에 대한 latent embedding은 여러개의 홉만큼 더 떨어진 노드에 대한 더 많은 정보를 가질 수 있게 된다.
    - 그래프에 있는 노드에 대해 latent embedding이 끝나면, latent embedding을 feed-forward network에 넣어서 classification / regression task를 진행할 수 있음
    - 아니면 모든 노드에 대한 embedding을 aggregate해서 전체 그래프에 대한 representation을 얻어서 graph classificication / regression을 진행할 수 있음
- node feature로 인해 노드간에 상관관계를 잘 capturing할 수 있기 때문에, traffic flow나 brain network에 사용됨

### 3-1-4 Recurrent Neural Network(RNN), Long Short-Term Memory(LSTM )Network

- RNN은 sequential 특성들을 인지하기 위해서 설계되었다. 
- 이전 pattern을 사용해서 next likely 시나리오를 predict하기도 한다.
- Xt는 input data, A는 network의 파라미터, ht는 학습된 hidden state을 의미한다.
- RNN의 가장 주된 문제점은 gradient 가 vanishing함에 따라서, short term memory 이슈가 발생한다는 것 → **이를 해결하기 위한 방안이 LSTM이다**
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1eZ8Be2RPr1pOm4f0uXH-s9eDVBPGEEy_){: width="80%" height="80%"}{: .center}
    
- LSTM은 인풋 데이터의 long-term dependencies를 학습할 수 있다.
- specially 설계된 메모리 유닛 덕분에 더 긴 historical info를 기억할 수 있다.
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1eTfLcy1TMJ184wJNRegnASoOChb3LMWg){: width="80%" height="80%"}{: .center}
    
    - LSTM 구성  : input gate, forget gate, output gate
    - 세개의 게이트를 통해서 새로운 input을 들여보낼 것인지, 중요하지 않은 정보를 무시할 것인지, 해당 정보를 출력에 영향을 미치게 할 것인지를 정함

### 3-1-5 Sequence to Sequence Model(Seq2Seq)

- 입출력의 길이가 고정된다. 
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1UGdHtddWsw8ZOG9KrXv15mJt6zQ4dc1t){: width="80%" height="80%"}{: .center}
    
    - Seq2Seq 구성 요소 : encoder, intermediate vector, decoder
    - sequence data간의 dependencies를 잘 포착하는 편

### 3-1-6 Autoencoder(AE)

- pass

## 3-2 Data Preparation and Deep Learning model Selection

- **STD instances는 적절한 data format으로 formulated된 이후 deep learning model에 입력되게 된다.**
- Trajectory와 time series는 sequence data 로 표현될 수 있지만, trajectory는 matrix으로 표현되어서 CNN이 spatial feature을 학습할 수 있도록 할 수도 있다.
- spatial map은 대부분 2D matrix으로 표현되는데, 가끔 graph으로 표현될 때도 있다.
    - 고속도로에 있는 traffic sensor들은 그래프로 표현 가능하다. (노드 = sensor, 엣지 = 두개의 이웃 센서를 잇는 road)
    - Graph CNN은 sensor graph를 process하고 future traffic을 predict하기 위해서 사용된다.
- spatial correlation을 반영하고 싶다면, GCNN을 사용하는 것이 좋은 선택이 될 수 있고, STD가 만약 image-like matrix으로 표현된다면 CNN이 적절한 선택일 것이라고 논문은 설명한다.
- **If the input is a sequence of image-like matrices, hybrid models that combine CNN and RNN such as ConvLSTM can be used [1], [61].**
    
    ![yes.JPG](http://drive.google.com/uc?export=view&id=1w2C3L9DTyZKAYSQfc4PhufP-hbHw-v6t){: width="80%" height="80%"}{: .center}
    

# 4 Deep Learning Models for Addressing STDM problems

- One can see the largest problemcategory is prediction, accounting for more than 70 percent works.
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1_r3F7yRAHHUsbrqtCLZzh4OAZSE_TAT3){: width="60%" height="60%"}{: .center}
    

## 4-1 Predictive Learning

- temporal 과 spatial correlation을 capture하기 위해서 CNN+RNN한 ConvLSTM이 제안된다.
- **ConvLSTM**
    - seq-to-seq prediction model임
    - input과 output 모두 spatial map matrix임
    - 각각의 layer은 ConvLSTM unit이고, convolutional operation을 input-to-state가 state-to-stat transition에 모두 적용
    - **(1) Cell state 란?**
        
        ![https://blog.kakaocdn.net/dn/mPRV9/btq4rdnodnB/sbgsr8Fb0UIkjmctMpQQs0/img.png](https://blog.kakaocdn.net/dn/mPRV9/btq4rdnodnB/sbgsr8Fb0UIkjmctMpQQs0/img.png)
        
        - cell state는 컨베이어 벨트와 같은 역할로 전체 체인을 지나면서, 일부 선형적인 상호작용만을 진행합니다. 정보가 특별한 변화 없이 흘러가는 것으로 이해하면 쉽습니다.
            
            LSTM의 핵심은 모델 구조도에서 위의 수평선에 해당하는 **cell state**입니다
            
        - LSTM에서 cell state를 업데이트하는 것이 가장 중요 한 일이며, **'gate**라는 구조가 cell state에 정보를 더하거나 제거하는 조절 기능을 합니다.
        >출처 : [https://hyen4110.tistory.com/25](https://hyen4110.tistory.com/25)
- **tried to apply ConvLSTM to other spatial map prediction tasks of different domains [1], [5], [24], [61], [64], [89], [143], [195]. [143] proposed a novel cross-city transfer learning method named RegionTrans for joint spatio-temporal data prediction among different cities.**
- Region Trans?
    - 여러개의 ConvLSTM layer을 포함하여 데이터 속에 숨어있는 spatio-temporal한 패턴을 catch함




[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
