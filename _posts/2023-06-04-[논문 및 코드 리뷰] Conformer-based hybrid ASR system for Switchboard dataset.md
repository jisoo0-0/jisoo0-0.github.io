---
layout: post
categories: 논문리뷰 코드리뷰
tags: audio, ASR
comments: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문 및 코드 리뷰] Conformer-based hybrid ASR system for Switchboard dataset"
---

# 1 논문 리뷰

> Zeineldeen, Mohammad, et al. "Conformer-based hybrid ASR system for Switchboard dataset." *ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE, 2022.
> 

## 1.1 논문 내용

### 1.1.0 Abstract

- Using a Conformer AM(Acoustic Models) for hybrid ASR system
- Study different training aspects and methods to improve word error rate & increasing training speed
- Time downsampling methods -> efficient training
- Transposed convolutions-> upsampling

→ 본 연구에서는 기존에 존재하였던 conformer 모델을 hybrid ASR system에 적용하고자 했음.

→ 특히 **word error rate은 줄이고 training speed는 향상시킬 수 있는 방법**에 대하여 연구하였으며 **time downsampling method**를 통해서 효과적인 훈련을 할 수 있었고 transposed convolution을 통해서 **output sequence를 upsampling**함

### 1.1.2 Introduction & Related Works

- Bidirectional Long Short-Term Memory (BLSTM)
    - 원래 관례적인 hybrid ASR system에 관한 음향 모델링에 자주 사용됨
- Time delay neural network, CNN
    - 최근 많이 연구되는 추세
- Self Attention Networks
    - 앞선 연구들보다 더 좋은 WER을 보임
- Conformer
    - SOTA 모델

→ Conformer의 성능은 우수하지만 아직까지 hybrid ASR system에 **Conformer가 Acoustic Model 으로 사용된 적이 없음**을 언급하며 이를 해결하고자 했음

→ self attention mechanis은 많은 메모리 자원을 요하고 이에 따라 **시간 복잡도가 sequence의 길이에 비례하여 기하급수적으로 증가**하기에 **time down sampling 기술**을 사용함

→ 더 나아가서 **효율적이지만 간단한 subsampling 방법을 소개**하고, 결과적으로 **더 좋은 training speed 와 memory consumption**을 보이게 함

### 1.1.3 Training Methods – Time down/up sampling

- 본 연구에서는 conformer based hybrid ASR model을 설계하기 위해서 다양한 training methods를 실험했음
- 기존 self attention mechanism은 메모리를 너무 많이 사용함
    - 전체 input batch가 할당된다면 당연히 training time이 증가될 것이기에 본 연구에서는 아래 figure과 같이 downsampling을 진행함
        - 하지만 다운샘플링을 하게된다면, frame 별로 계산되는 loss objective function과 target의 개수가 달라지게 되고 이를 해결하기 위하여 transpose convolution을 사용해서 upsampling을 적용해줌
            
            ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/119ad286-8407-41b0-bd91-bde908c93f33){: width="100%" height="100%"}{: .center}
            

### 1.1.4 Training Methods – ETC

- 기타 학습 기법
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/3b8f3d97-d908-485c-bbc6-b146aae8e59c){: width="100%" height="100%"}{: .center}
    
    - intermediate loss
        - 이를 사용해서 training stability를 높이려고 함
        - 이는 intermediate loss를 각기다른 layer사이에 추가하는 것임
        - transposed conv layer는 upsampling의 intermediate loss를 계산하기 위하여 사용됨
    - Parameter sharing
        - model size를 줄이기 위해서 intermediate loss layer과 transposed convolution 의 파라미터가 각각 공유됨
    - Long Skip
        - 학습 feature을 다시 사용하기 위한 방법으로 Long skip이 사용되었는데 , 본 연구에서는 VGG network 의 output을 각각의 conformer block의 input과 연결해 주며 이를 해결함
    - Focal Loss
        - Cross entropy objective function을 reshape하며 잘 분류된 타겟에 가중치를 줄여서 misclassified 된 target에 focusing 하기 위한 방법임
    - Spec Augement
        - 음성 인식에서 널리 사용되고 있는 음성 증강 기법으로, 본 연구에서는 주파수 채널과 블럭을 masking하며 데이터를 증강시킴
    - Sequence discriminative training
        - training과 recognition사이의 불일치를 줄여주어, 모델의 성능을 높여줌

### 1.1.5 Experimental Setup

- **RASR을 통해서 feature extraction과 recogniton을 진행함**
- **RETURNN을 통해서 acoustic model을 training함**

## 1.2 데이터셋

- 데이터셋 이름
    - Hub5’00
- 데이터셋 구성
    - Switchboard(SWB)
    - CallHome(CH)
- 데이터셋 설명
    - Speech Recognition에 사용되는 데이터셋으로, 11시간 가량의 영어 통화 녹음 데이터로 이루어짐
    - Linguistic Data Consortium(LDC) 가 제공한 데이터셋으로 documetation table은 speech segment에 대한 information이 담겨져 있음
    - transcripts 또한 제공됨
- 데이터셋 사용
    - 본 연구에서는 SWB 를 사용함
- 데이터셋 링크
    - [https://catalog.ldc.upenn.edu/LDC2002S09](https://catalog.ldc.upenn.edu/LDC2002S09)

# 2 코드 리뷰

- It’s overall architecture
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/d26606c7-7879-4fd3-883a-df18fa755313){: width="100%" height="100%"}{: .center}
    

## 2.1 SpecAug

> SpecAugment [24] is a data augmentation method that **masks out blocks of frequency channels and blocks of time steps**.

- Spec Aug 위치
![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/d70e4b6e-c0d7-4b0b-aa9b-17f464c5dc20){: width="100%" height="100%"}{: .center}
    
- Spec Aug 코드 구현
    
    ```python
    #/baseline_12l.config 파일의 82번째 줄~
    def transform(data, network, time_factor=1):
      x = data.placeholder
      from returnn.tf.compat import v1 as tf
      # summary("features", x)
      step = network.global_train_step
      step1 = tf.where(tf.greater_equal(step, 1000), 1, 0)
      step2 = tf.where(tf.greater_equal(step, 2000), 1, 0)
      def get_masked(): # 해당함수를 통해서 spec augment 부분이 진행됨 
          x_masked = x # random mask 부분을 계속 업데이트 시켜주는 형식으로 코드가 진행되는데 처음에는 time 축에 maksing을 진행하고
          x_masked = random_mask(
            x_masked, batch_axis=data.batch_dim_axis, axis=data.time_dim_axis,
            min_num=step1 + step2, max_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // 100, 2) * (1 + step1 + step2 * 2),
            max_dims=20 // time_factor) 
          x_masked = random_mask( #이 부분에서 feaure 즉 channel 축의 masking이 진행된다. 
            x_masked, batch_axis=data.batch_dim_axis, axis=data.feature_dim_axis,
            min_num=step1 + step2, max_num=2 + step1 + step2 * 2,
            max_dims=data.dim // 5)
          #summary("features_mask", x_masked)
          return x_masked
      x = network.cond_on_train(get_masked, lambda: x)
      return x
    ```
    

## 2.2 VGG Block

> For NN training, we use 40-dimensional Gammatone features [28]. The first block of the NN consists of a VGG network similar to [19]. We use **4 convolution layers each having 3×3 kernel size**. The number of output filters for each layer are 32, 64, 64, 32 respectively. **We apply Swish activation** [29] between all convolution layers which we observe to be better than using ReLU [30]. Moreover, we apply **max-pooling layer over feature dimension between first and second convolution layers**. The last convolution layer is a **strided convolution used for time downsampling by factor of 3**. This is followed by 12 conformer blocks. For time upsampling, a transposed convolution is used. The attention dimension of each MHSA module is 512 with 8 attention heads. The dimension of the feed-forward module is 2048. We also use relative positional encoding. The output labels consists of state-tied triphones using CART [31]

- VGG Block 위치
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/712d7974-cc94-4630-9677-af3ba54927af){: width="100%" height="100%"}{: .center}
    
- VGG Block 코드 구현
    
    ```python
    #/baseline_12l.config 파일의 1669번째 줄~
    'source': { 'class': 'eval', 
                  'eval': "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)", #SpecAugment 가 적용된 데이터를 불러와서
                  'from': 'data'},
    'source0': {'axis': 'F', 'class': 'split_dims', 'dims': (-1, 1), 'from': 'source'}, #차원 조절을 해주고
    
    #/baseline_12l.config 파일의 167번째 줄~
    'c1': {'class': 'conv', 'filter_size': (3, 3), 'from': 'source0', 'n_out': 32, 'padding': 'same', 'with_bias': True}, #(그림에서 1번 부분) filter_size 3,3인 conv 에 입력함 
    'c2': {'class': 'conv', 'filter_size': (3, 3), 'from': 'y4', 'n_out': 32, 'padding': 'same', 'strides': (3, 1), 'with_bias': True},#(그림에서 5번 부분) stride convolution 적용
    'c3': {'class': 'conv', 'filter_size': (3, 3), 'from': 'p1', 'n_out': 64, 'padding': 'same', 'with_bias': True}, #(그림에서 3번 부분) max pooling 된 값을 들고와서 convolution 연산 적용
    'c4': {'class': 'conv', 'filter_size': (3, 3), 'from': 'y3', 'n_out': 64, 'padding': 'same', 'with_bias': True}, #(그림에서 4번 부분)위의 코드에서 나온 값을 아래 1716~줄의 코드를 통해 swish activation을 적용해주고 다시 convolution을 진행함. 
    
    #/baseline_12l.config 파일의 1716번째 줄~
    #차원 합쳐주기 
    'vgg_conv_merged': {'axes': 'static', 'class': 'merge_dims', 'from': 'y2'}, #VGG block에서 생성된 데이터를 Linear 레이어로 옮겨주기 위한 것
    #각각 cn을 불러와서 activation fuction인 swish를 적용해주는 코드임. activation은 conv 뒤에 항상 적용되어서 이렇게 코드가 작성된 듯 함. 
    'y1': {'activation': 'swish', 'batch_norm': False, 'class': 'activation', 'from': 'c1'},
    'y2': {'activation': 'swish', 'batch_norm': False, 'class': 'activation', 'from': 'c2'}, 
    'y3': {'activation': 'swish', 'batch_norm': False, 'class': 'activation', 'from': 'c3'},
    'y4': {'activation': 'swish', 'batch_norm': False, 'class': 'activation', 'from': 'c4'}} 
    
    #/baseline_12l.config 파일의 1664번째 줄~
    'p1': {'class': 'pool', 'from': 'y1', 'mode': 'max', 'padding': 'same', 'pool_size': (1, 2)}, #(그림에서 2번 부분) y1 즉 swish 가 적용된 feature들에 max pooling을 시켜주고, 
    'p2': {'class': 'pool', 'from': 'y4', 'mode': 'max', 'padding': 'same', 'pool_size': (1, 2)}, 
    
    ```
    

## 2.3 Linear and Dropout

> These layers consist of a transposed convolution for upsampling followed by an MLP of one **linear projection layer** **with dimension 512 × 512**. …. We apply **dropout of 10%** for all conformer modules as well as embedding and attention dropout.

- Linear layer와 dropout 의 위치
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/0dae575c-ce44-4196-93c0-48722c1c129d){: width="100%" height="100%"}{: .center}
    
- Linear layer와 dropout 코드 구현
    
    ```python
    #/baseline_12l.config 파일의 1716번째 줄~
    #차원 합쳐주기 
    'vgg_conv_merged': {'axes': 'static', 'class': 'merge_dims', 'from': 'y2'}, #VGG block에서 생성된 데이터를 Linear 레이어로 옮겨주기 위한 것
    
    #/baseline_12l.config 파일의 1670번째 줄~
    'source_dropout': {'class': 'dropout', 'dropout': 0.1, 'from': 'source_linear'}, #linear 가 적용된 이후 dropout을 
    'source_linear': { 'activation': None,
                       'class': 'linear', #linear 함수를 적용해줌
                       'forward_weights_init': {'class': 'VarianceScaling', 'distribution': 'uniform', 'mode': 'fan_in', 'scale': 0.78},
                       'from': 'vgg_conv_merged', #이전 part에서 생성된 값들을 전달 받았다는 부분
                       'n_out': 512, #512*512으로 linear output 설정
                       'with_bias': False},
    ```
    

## 2.3 N Conformer Block - 1

> The standard conformer architecture [10] consists mainly of four modules: **feed-forward module (FFN)**, multi-head self-attention module **(MHSA)**, convolution module (**Conv**), and **another feed-forward module**. ConformerBlocki = **LayerNorm(xF F N2 ) …** The attention dimension of each MHSA module is 512 with 8 attention heads. The dimension of the feed-forward module is 2048.

- Conformer Block 위치
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/750fbc5a-8641-4099-8e2f-e360e9597ada){: width="100%" height="100%"}{: .center}
    
- Conformer Block 코드 구현
    
    ### 2.3.1 1/2 FFN -1
    
    ```python
    #/baseline_12l.config 파일의 231번째 줄~
    'conformer_block_01_ffmod_1_ln': {'class': 'layer_norm', 'from': 'source_dropout'}, #dropout 이 적용된 데이터들을 기반으로 layer norm 을 적용해줌
    #/baseline_12l.config 파일의 212번째 줄~
    'conformer_block_01_ffmod_1_ff1': { 'activation': None, #(그림에서 1번 부분) half FFN 부분 처음 적용
                                        'class': 'linear',
                                        'forward_weights_init': { 'class': 'VarianceScaling',
                                                                  'distribution': 'uniform',
                                                                  'mode': 'fan_in',
                                                                  'scale': 0.78},
                                        'from': 'conformer_block_01_ffmod_1_ln',
                                        'n_out': 2048,
                                        'with_bias': True},
    #/baseline_12l.config 파일의 236번째 줄~
    'conformer_block_01_ffmod_1_swish': {'activation': 'swish', 'class': 'activation', 'from': 'conformer_block_01_ffmod_1_ff1'}, #FFN뒤에 적용되는 swish 함수
    #/baseline_12l.config 파일의 210번째 줄~
    'conformer_block_01_ffmod_1_drop1': {'class': 'dropout', 'dropout': 0.1, 'from': 'conformer_block_01_ffmod_1_swish'}, #dropout 0.1 적용
    #/baseline_12l.config 파일의 221번째 줄~
    'conformer_block_01_ffmod_1_ff2': { 'activation': None,  #MHSA module의 attention 차원이 512라서 조정해주는 부분
                                          'class': 'linear',
                                          'forward_weights_init': { 'class': 'VarianceScaling',
                                                                    'distribution': 'uniform',
                                                                    'mode': 'fan_in',
                                                                    'scale': 0.78},
                                          'from': 'conformer_block_01_ffmod_1_drop1',
                                          'n_out': 512,
                                          'with_bias': True},
    #/baseline_12l.config 파일의 211번째 줄~
    'conformer_block_01_ffmod_1_drop2': {'class': 'dropout', 'dropout': 0.1, 'from': 'conformer_block_01_ffmod_1_ff2'},
    #/baseline_12l.config 파일의 230번째 줄~
    'conformer_block_01_ffmod_1_half_step': {'class': 'eval', 'eval': '0.5 * source(0)', 'from': 'conformer_block_01_ffmod_1_drop2'},#(그림에서 1번 부분) 이 끝나는 부분으로,  broadcasted extension이 적용됨 
    
    ```
    
    ### 2.3.2 MHSA
    
    ```python
    #/baseline_12l.config 파일의 232번째 줄~
    'conformer_block_01_ffmod_1_res': { 'class': 'combine', #(그림에서 1번에서 2번으로 넘어가는 부분) add 해주는 부분 (아마 residual connection을 의미하려고 res을 쓴 것 같음)
                                          'from': ['conformer_block_01_ffmod_1_half_step', 'source_dropout'],
                                          'kind': 'add',
                                          'n_out': 512},
    
    #/baseline_12l.config 파일의 283번째 줄~
    'conformer_block_01_self_att_ln': {'class': 'layer_norm', 'from': 'conformer_block_01_ffmod_1_res'},  #self atention layer전에 layer normalization 해주는 부분
    
    #/baseline_12l.config 파일의 265번째 줄~ #(그림에서 2번 부분)
    'conformer_block_01_self_att': { 'attention_dropout': 0.1, #drop out 적용
                                       'class': 'self_attention',#(그림에서 1번부분)
                                       'forward_weights_init': {'class': 'VarianceScaling', 'distribution': 'uniform', 'mode': 'fan_in', 'scale': 0.78},
                                       'from': 'conformer_block_01_self_att_ln',
                                       'key_shift': 'conformer_block_01_self_att_ln_rel_pos_enc',
                                       'n_out': 512, #차원 512
                                       'num_heads': 8, #self attention의 head 개수 8
                                       'total_key_dim': 512}, #Key Query Value에서의 key차원을 설정해줌
    
    #/baseline_12l.config 파일의 274번째 줄~
    '': { 'activation': None, #linear 적용해준 파트 
                                              'class': 'linear', 
                                              'forward_weights_init': { 'class': 'VarianceScaling',
                                                                        'distribution': 'uniform',
                                                                        'mode': 'fan_in',
                                                                        'scale': 0.78},
                                              'from': 'conformer_block_01_self_att',
                                              'n_out': 512,
                                              'with_bias': False},
    
    #/baseline_12l.config 파일의 284번째 줄~ 
    'conformer_block_01_self_att_ln_rel_pos_enc': { 'class': 'relative_positional_encoding', #attention mechanism 의 positional encoding부분임.
                                                      'forward_weights_init': { 'class': 'VarianceScaling',
                                                                                'distribution': 'uniform',
                                                                                'mode': 'fan_in',
                                                                                'scale': 0.78},
                                                      'from': 'conformer_block_01_self_att_ln',
                                                      'n_out': 64},
    
    #/baseline_12l.config 파일의 273번째 줄~ 
    'conformer_block_01_self_att_dropout': {'class': 'dropout', 'dropout': 0.1, 'from': 'conformer_block_01_self_att_linear'}, #dropout 적용
    
    #/baseline_12l.config 파일의 291번째 줄~
    'conformer_block_01_self_att_res': { 'class': 'combine', #self attention module이 적용된 최종적인 값 그림에서 2번~3번 넘어갈 때 사용됨
                                           'from': ['conformer_block_01_self_att_dropout', 'conformer_block_01_ffmod_1_res'],
                                           'kind': 'add',
                                           'n_out': 512},
    
    ```
    
    ### 2.3.3 Conv Module
    
    ```python
    
    #/baseline_12l.config 파일의 193번째 줄~
    'conformer_block_01_conv_mod_ln': {'class': 'layer_norm', 'from': 'conformer_block_01_self_att_res'}, #self attention이 적용된 최종적인 값에 layer norm을 적용해줌
    
    #/baseline_12l.config 파일의 187번째 줄~
    'conformer_block_01_conv_mod_pointwise_conv1': { 'activation': None, #linear layer 적용 및 output 차원을 1024으로 조정
                                                       'class': 'linear',
                                                       'forward_weights_init': { 'class': 'VarianceScaling',
                                                                                 'distribution': 'uniform',
                                                                                 'mode': 'fan_in',
                                                                                 'scale': 0.78},
                                                       'from': 'conformer_block_01_conv_mod_ln',
                                                       'n_out': 1024,
                                                       'with_bias': True},
    
    #/baseline_12l.config 파일의 182번째 줄~
    #glu : 부록 참고
    'conformer_block_01_conv_mod_glu': { 'activation': 'identity', #항등함수 사용
                                           'class': 'gating',
                                           'from': 'conformer_block_01_conv_mod_pointwise_conv1',
                                           'gate_activation': 'sigmoid'}, #gate의 activation은 sigmoid으로 설정
    
    #/baseline_12l.config 파일의 173번째 줄~
    #depthwise convolution :  각 단일 채널에 대해서만 수행되는 필터들을 사용
    'conformer_block_01_conv_mod_depthwise_conv2': { 'activation': None,
                                                       'class': 'conv',
                                                       'filter_size': (8,),
                                                       'from': 'conformer_block_01_conv_mod_glu',
                                                       'groups': 512,
                                                       'n_out': 512,
                                                       'padding': 'same', #같은 값으로 패딩해줌
                                                       'with_bias': True},
    
    #/baseline_12l.config 파일의 172번째 줄~
    #batch normalization진행
    'conformer_block_01_conv_mod_bn': {'class': 'batch_norm', 'from': 'conformer_block_01_conv_mod_depthwise_conv2'},
    
    #/baseline_12l.config 파일의 209번째 줄~
    #conv 뒤에 오는 swish activation 진행
    'conformer_block_01_conv_mod_swish': {'activation': 'swish', 'class': 'activation', 'from': 'conformer_block_01_conv_mod_bn'},
    
    #/baseline_12l.config 파일의 196번째 줄~
    #pointwise convolution : 공간방향의 컨볼루션은 진행하지 않고 각 채널 방향의 컨볼루션을 진행하는 것
    'conformer_block_01_conv_mod_pointwise_conv2': { 'activation': None,
                                                       'class': 'linear',
                                                       'forward_weights_init': { 'class': 'VarianceScaling',
                                                                                 'distribution': 'uniform',
                                                                                 'mode': 'fan_in',
                                                                                 'scale': 0.78},
                                                       'from': 'conformer_block_01_conv_mod_swish',
                                                       'n_out': 512,
                                                       'with_bias': True},
    
    #/baseline_12l.config 파일의 181번째 줄~
    #dropout layer임
    'conformer_block_01_conv_mod_drop': {'class': 'dropout', 'dropout': 0.1, 'from': 'conformer_block_01_conv_mod_pointwise_conv2'},
    
    #/baseline_12l.config 파일의 205번째 줄~ 
    'conformer_block_01_conv_mod_res': { 'class': 'combine', #앞에서 설정된 MHSA과 Conv module을 합쳐주는 residual connection 부분임. 그림에서 3번~4번 넘어가는 부분이라고 이해하면 됨
                                           'from': ['conformer_block_01_conv_mod_drop', 'conformer_block_01_self_att_res'],
                                           'kind': 'add',
                                           'n_out': 512},
    
    ```
    
- Layer Norm : [https://returnn.readthedocs.io/en/latest/layer_reference/norm.html](https://returnn.readthedocs.io/en/latest/layer_reference/norm.html)
    
    ### 2.3.4 1/2 FFN - 2
    
    ```python
    #/baseline_12l.config 파일의 205번째 줄~  #Convolution Module이 적용된 값들을 들고와서 layer normalization을 적용시켜줌.
    'conformer_block_01_ffmod_2_ln': {'class': 'layer_norm', 'from': 'conformer_block_01_conv_mod_res'},
    
    #/baseline_12l.config 파일의 239번째 줄~ 
    'conformer_block_01_ffmod_2_ff1': { 'activation': None, 
                                          'class': 'linear', #Linear layer으로 output dim을 2048으로 조정해줌
                                          'forward_weights_init': { 'class': 'VarianceScaling',
                                                                    'distribution': 'uniform',
                                                                    'mode': 'fan_in',
                                                                    'scale': 0.78},
                                          'from': 'conformer_block_01_ffmod_2_ln',
                                          'n_out': 2048,
                                          'with_bias': True},
    
    #/baseline_12l.config 파일의 263번째 줄~  
    'conformer_block_01_ffmod_2_swish': {'activation': 'swish', 'class': 'activation', 'from': 'conformer_block_01_ffmod_2_ff1'}, #swish activation 적용
    
    #/baseline_12l.config 파일의 237번째 줄~ 
    'conformer_block_01_ffmod_2_drop1': {'class': 'dropout', 'dropout': 0.1, 'from': 'conformer_block_01_ffmod_2_swish'},#dropout 적용
    
    #/baseline_12l.config 파일의 248번째 줄~ 
    'conformer_block_01_ffmod_2_ff2': { 'activation': None,
                                          'class': 'linear', #Linear layer으로 output dim을 512로 조정해줌
                                          'forward_weights_init': { 'class': 'VarianceScaling',
                                                                    'distribution': 'uniform',
                                                                    'mode': 'fan_in',
                                                                    'scale': 0.78},
                                          'from': 'conformer_block_01_ffmod_2_drop1',
                                          'n_out': 512,
                                          'with_bias': True},
    
    #/baseline_12l.config 파일의 238번째 줄~ 
    'conformer_block_01_ffmod_2_drop2': {'class': 'dropout', 'dropout': 0.1, 'from': 'conformer_block_01_ffmod_2_ff2'},#dropout 적용
    
    #/baseline_12l.config 파일의 257번째 줄~ 
    'conformer_block_01_ffmod_2_half_step': {'class': 'eval', 'eval': '0.5 * source(0)', 'from': 'conformer_block_01_ffmod_2_drop2'}, #(그림에서 4번 부분) 이 끝나는 부분으로,  broadcasted extension이 적용됨 
    
    #/baseline_12l.config 파일의 259번째 줄~  
    #그림 4번에서 5번으로 넘어가는 과정의 residual connection 을적용해준 부분으로 convolution module을 거친 값과 1/2 FFN을 거친 값을 connecting시켜줌
    'conformer_block_01_ffmod_2_res': { 'class': 'combine',
                                          'from': ['conformer_block_01_ffmod_2_half_step', 'conformer_block_01_conv_mod_res'],
                                          'kind': 'add',
                                          'n_out': 512},
    
    ```
    
    - eval layer : [https://returnn.readthedocs.io/en/latest/layer_reference/custom.html](https://returnn.readthedocs.io/en/latest/layer_reference/custom.html)
    
    ### 2.3.5 layer norm
    
    ```python
    #/baseline_12l.config 파일의 264번째 줄~ 
    #그림에서 5번 부분을 나타내는 코드로, layer normalization을 적용해 주는 부분임.  
    'conformer_block_01_ln': {'class': 'layer_norm', 'from': 'conformer_block_01_ffmod_2_res'},
    ```
    
    ### 2.3.6 conformer block 쌓는 부분
    
    ```python
    #/baseline_12l.config 파일의 171번째 줄~ 
    'conformer_block_01': {'class': 'copy', 'from': 'conformer_block_01_ln'}, #이전 part에서 최송 생성된 conformer block 을 copy해서 conformer_block_01에 저장
    
    #/baseline_12l.config 파일의 1680번째 줄~ 
    #conformer block을 transposed convolution residual connection하여 전달함. 
    'transposed_conv_1_res': {'class': 'combine', 'from': ['source_linear', 'conformer_block_01'], 'kind': 'add', 'n_out': 512},
    
    #/baseline_12l.config 파일의 1677번째 줄~ 
    #transposed_conv_1_res 이외 12개의 conformer block을 연결해서 최종적으로 transposed_conv_12_res 이 완성됨. 
    #transposed_conv_n_res는 각각의 conformer_block_n의 최종본을 source linear와 residual connection한 결과값으로
    #이들을 모으면 conformer block구간이 끝난다고 이해할 수 있음.
    'transposed_conv_10_res': {'class': 'combine', 'from': ['source_linear', 'conformer_block_10'], 'kind': 'add', 'n_out': 512},
    'transposed_conv_11_res': {'class': 'combine', 'from': ['source_linear', 'conformer_block_11'], 'kind': 'add', 'n_out': 512},
    'transposed_conv_12_res': {'class': 'combine', 'from': ['source_linear', 'conformer_block_12'], 'kind': 'add', 'n_out': 512},
    'transposed_conv_1_res': {'class': 'combine', 'from': ['source_linear', 'conformer_block_01'], 'kind': 'add', 'n_out': 512},
    'transposed_conv_2_res': {'class': 'combine', 'from': ['source_linear', 'conformer_block_02'], 'kind': 'add', 'n_out': 512},
    'transposed_conv_3_res': {'class': 'combine', 'from': ['source_linear', 'conformer_block_03'], 'kind': 'add', 'n_out': 512},
    'transposed_conv_4_res': {'class': 'combine', 'from': ['source_linear', 'conformer_block_04'], 'kind': 'add', 'n_out': 512},
    'transposed_conv_5_res': {'class': 'combine', 'from': ['source_linear', 'conformer_block_05'], 'kind': 'add', 'n_out': 512},
    'transposed_conv_6_res': {'class': 'combine', 'from': ['source_linear', 'conformer_block_06'], 'kind': 'add', 'n_out': 512},
    'transposed_conv_7_res': {'class': 'combine', 'from': ['source_linear', 'conformer_block_07'], 'kind': 'add', 'n_out': 512},
    'transposed_conv_8_res': {'class': 'combine', 'from': ['source_linear', 'conformer_block_08'], 'kind': 'add', 'n_out': 512},
    'transposed_conv_9_res': {'class': 'combine', 'from': ['source_linear', 'conformer_block_09'], 'kind': 'add', 'n_out': 512},
    
    ```
    

### 부록) basic positional encoding VS relative positional encoding

- basic positional encoding
    - 해당 token이나 value의 절대적인 위치값을 인코딩함
    
    ex) Wanna go starbucks?
    
    [[0,1,2],
    
    [0,1,2],
    
    [0,1,2]] 가 position encoding 으로 들어감 
    
- relative positional encoding
    - 해당 token이나 값을 기준으로하는 상대적인 위치값을 인코딩함
    - T5 모델에서는 특정 거리 이상에서는 상대적인 위치에 대한 정확한 값이 중요하지 않을 수 있으며, 멀리 떨어진 토큰에 대한 위치정보는 덜 중요하게 반영할수 있도록 함
    
    ex) Wanna go starbucks?
    
    [[0,1,2],
    
    [-1,0,1],
    
    [-2,-1,0]] 가 position encoding 으로 들어감 
    

### 부록) GLU (Gated Linear Unit)

> Dauphin, Yann N., et al. "Language modeling with gated convolutional networks." *International conference on machine learning*. PMLR, 2017.

- LSTM으로 유사한 gate임
    - LSTM과는 다르게 forget gate가 따로 없음.
- 언어 모델을 위한 모델이었기 때문에 미래 단어 정보를 zeropedding시켜줘야함.
    - 미래 정보를 알고있으면 예측하는 의미가 없기 때문임
- input sentence가 input으로 들어오면 각 word들은 Lookup table에 있는 vector embedding으로 represented됨
- 각 layer의 output은 linear projection에 sigmoid를 적용함
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/90c8d0e9-8b9f-4063-af0f-6ec4527b9b74){: width="70%" height="70%"}{: .center}
    

## 2.4 N Conformer Block - 2

> We add two **intermediate loss layers** on the output of the 4 th and 8 th conformer blocks.
> 
- 4, 8번째 conformer block 코드 구현
    
    ```python
    #/baseline_12l.config 파일의 1697번째 줄~ 
    #그림에서 1번 부분임
    'transposedconv_4': { 'L2': 0.01, #이 부분이 다른 conformer block에는 없고,4번째 8번째 conformer block에만 있음
                            'activation': 'swish',
                            'class': 'transposed_conv',
                            'dropout': 0.1,
                            'filter_size': [3],
                            'from': 'transposed_conv_4_res',
                            'n_out': 512,
                            'reuse_params': 'transposedconv', #transposed convolution의 param을 재사용함
                            'strides': [3]},
    
    #/baseline_12l.config 파일의 1661번째 줄~ 
    #reinterpret data 으로 설정해줌. 이부분이 conformer block 4,8,12에만 존재함-> loss 설정과 관련있는듯
    'masked_tconv_4': {'class': 'reinterpret_data', 'from': 'transposedconv_4', 'size_base': 'data:classes'},
    
    #/baseline_12l.config 파일의 151번째 줄~ 
    #masked_tconv_4 를 aux_output_block과 연결해줌
    'aux_output_block_4': {'class': 'copy', 'from': 'masked_tconv_4'},
    
    #/baseline_12l.config 파일의 139번째 줄~ 
    #그림에서 2번 부분임
    #aux multi layer perceptron block을 생성해줌
    
    network = { 'aux_MLP_block_4': { 'activation': None,
                           'class': 'linear',
                           'forward_weights_init': {'class': 'VarianceScaling', 'distribution': 'uniform', 'mode': 'fan_in', 'scale': 0.78},
                           'from': 'aux_output_block_4',
                           'n_out': 512,
    
    #/baseline_12l.config 파일의 1661번째 줄~ 
    'aux_output_block_4_ce': { 'class': 'softmax', #(그림에서 3번 부분) softmax으로 출력함수 설정
                                 'dropout': 0.0,
                                 'from': 'aux_MLP_block_4',
                                 'loss': 'ce', #(그림에서 4번 부분) loss를 cross entropy 으로 설정해줌. 
                                 'loss_opts': {'focal_loss_factor': 2},
                                 'loss_scale': 0.3,
                                 'target': 'classes'},
    ```
    

- reinterpret data layer : [https://returnn.readthedocs.io/en/latest/layer_reference/shape.html#reinterpret-data-layer](https://returnn.readthedocs.io/en/latest/layer_reference/shape.html#reinterpret-data-layer)

## 2.5 Transposed Conv, Softmax and CE Loss

> … we use transposed convolution [18] inspired from computer vision in order to upsample again to the **frame-wise target alignment length**. For consistency, the **filter size and the stride of the transposed convolution are set to the time reduction factor**.We apply **weight decay** [35] with a value of **0.01** to the transposed convolution layers. To avoid overfitting, we use **focal loss** [23] with a **factor of 2**. At the **final output layer**, we use **CE loss** **smoothing** with a factor of **0.1**.
 
- Transposed Conv, Softmax and CE Loss 위치
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/72643033-64e1-4d92-8e82-86becb5cb1c5){: width="100%" height="100%"}{: .center}
    
- Transposed Conv, Softmax and CE Loss 코드 구현
    
    ```python
    #/baseline_12l.config 파일의 1659번째 줄~ 
    #합쳐진 conformer block들을 encoder으로 선언해주는 부분(복사해줌)
    'encoder': {'class': 'copy', 'from': 'transposed_conv_12_res'},
    
    #/baseline_12l.config 파일의 1689번째 줄~ 
    #(그림에서 1번 부분) encoder을 기반으로 transposed convolution을 선언해주는 부분임
    'transposedconv': { 'L2': 0.01,#Weight decay를 0.01으로 설정
                        'activation': 'swish',#활성화는 swish
                        'class': 'transposed_conv',
                        'dropout': 0.1, #dropout 비율 0.1
                        'filter_size': [3], #filter 사이즈
                        'from': 'encoder',
                        'n_out': 512,
                        'strides': [3]},#stride 사이즈
    
    #/baseline_12l.config 파일의 1660번째 줄~ 
    #masking 시켜주는 부분
    'masked_tconv': {'class': 'reinterpret_data', 'from': 'transposedconv', 'size_base': 'data:classes'},
    
    #/baseline_12l.config 파일의 1663번째 줄~ 
    # output layer 설정해주는 부분
    'output': {'class': 'softmax', (그림에서 2번 부분) 
    						'dropout': 0.0, 
    						'from': 'masked_tconv',
    						'loss': 'ce', (그림에서 3번 부분) loss를 cross entropy loss으로 설정해줌
    						'loss_opts': {'focal_loss_factor': 2},  #focal loss factor을 2로 설정
    						'target': 'classes'},
    
    ```
    

### 부록) Smoothing 하는 이유

- Smoothing을 하는 이유는 각 출력값을 비슷한 분포로 맞춰주기 위해서임
- 정말 나이브하게 생각하면 입력값이 [0.01,0.9,0.2] 이라고 했을때 해당 상태에서 그냥 softmax를 적용한다면**[0.2153, 0.5243, 0.2604]** 값이 나오지만 /4를 한채 softmax를 취하면 **[0.3032, 0.3788, 0.3180]**값이 나오는 것을 확인할 수 있음
    - 즉 한쪽에 값이 몰리는 것을 방지해줄 수 있고 학습이 잘 되도록 유도할 수 있음
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/dd2eaf4e-ca73-4d2b-9ab4-ee90a2f8c8b5){: width="100%" height="100%"}{: .center}


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