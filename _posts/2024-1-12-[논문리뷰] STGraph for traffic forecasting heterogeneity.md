---
layout: post
categories: 개념정리
tags: gcn, STGCN, dynamic_gcn
comments: true
math: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] Learning dynamics and heterogeneity of spatial-temporal graph data for traffic forecasting"
---
출처
> Guo, Shengnan, et al. "Learning dynamics and heterogeneity of spatial-temporal graph data for traffic forecasting." IEEE Transactions on Knowledge and Data Engineering 34.11 (2021): 5415-5428.

Copyright of figures and other materials in the paper belongs to original authors.

# 0 Abstract

- 이 연구 왜했나?
    - 정확한 traffic forcasting을 위해서 시공간 정보를 이용하여 장기적이면서도 성능 좋은 모델을 제안하기 위하여
- temporal 차원에서는 self-attention 방법을 통해서 local context를 사용할 수 있도록 함
    - global receptive field를 사용할 수 있다는 늬앙스를 풍김
        - global receptive field는 일부 범위가 아니라 전체적인 범위에 대한 수용 범위를 말함
- spatial 차원에서는 dynamic graph conv module을 개선해서 self attention을 사용함
    - dynamic 방식에 있는 공간 상관관계를 포착하기 위해서 self attention을 사용

# 1 Introduction

- 본 연구의 의의
    - self attention을 사용한 교통 예측 모델을 제안함
    - trend에 aware한 self-attention 모듈을 제안하였고, 이는 local context에 주의를 줄 수 있음
    - dynamic graph conv module을 개선해서 dynamic manner에 spatial correlation을 모델링함
    - 기존 연구와는 다르게 명시적으로 traffic data의 주기성과 공간의 계층정보를 반영함으로서 모델의 성능을 개선함
- ASTGCN에 비해서 개선된 점들
    - **1D-CNNS을 사용하는대신 Temporal Trend-Aware Multi Head Self Attention을 사용**
        - traffic data의 dynamic한 정보를 좀 더 잘 반영할 수 있음
    - **새로운 dynamic graph conv 모듈을 design함**
        - spatial correlation을 dynamic한 방법으로 capture할 수 있음
    - spatial적으로 계층적인 요소를 명시적으로 모델링해서 모델의 성능을 높임

# 2 Related Works

## 2.2 Attention Mechanism

- It transform a sequence by using selfattention to compute a series of context-informed vector representations of the symbols in the input sequence. As each symbol’s representation is directly informed by all other symbols’ representations, **this results in an effective global receptive field,** which stands in contrast to RNNs and CNNs. Hence, self-attention offers a more flexible mechanism to model the **complex dynamics and long-term patterns of traffic data.**
    - 이렇게 transformer을 설명한 이유가, 전체적으로 Spatio Temporal 한 정보를 받아들일 수 있는 GCN에 self attention을 사용해 주었기 때문에 global receptive field, long term pattern의 동적인 부분을 capture할 수 있다는 장점을 가져왔다고 주장하기 위해서인것 같음.

# 3 Problem Statement and Preliminaries

## 3.1 Problem Statement

### Definition1 - Traffic Network

- 무방향 그래프 G = (V,E)로 설정
- 본 연구에서 제안된 모델은 무방향/양방향 모두 다 적용가능하다고 설명함

### Definition2 - Traffic Signal Matrix

- time slice t에서의 traffic network G는 trafic signal matrix으로 표현 가능함.

### Problem Statement

- global periodic sequence랑 local periodic sequence를 참고해서 future traffic signal matric을 예측하려고 함.
- 이전 연구들은 그냥 X(과거 traffic signal)만 사용해서 예측을 진행했기 때문에, 모든 정보를 사용한 모델을 ASTGNN(p)으로 denote함

## 3.2 Attention Mechanism

- query와 key/value pair을 mapping하는 것을 아웃풋으로 가짐.
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/bbedc170-8a61-465f-bf32-440681a76144){: width="50%" height="50%"}{: .center}
    
    - output은 value들의 가중합으로, 가중치는 key와 query에 따른 각각의 value에 따라서 상이해짐.
    - Scaled dot product attention
        - weight가 query와 value 사이의 dot product으로 계산된 attention function임
        - 

# 4 Attention based spatial-temporal graph neural networks

- ASTGNN 은 기본적인 encoder-decoder framework에 기반함
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/76bed150-e5b5-4b98-839b-f9217795bc11){: width="50%" height="50%"}{: .center}
    
    - model이 깊어져도 효과적인 학습이 가능하게 하도록, residual connection과 layer normalization이 block 속에서 진행되었음.
- Pipeline설명
    - encoder으로 들어온 input sequence X는 decoder으로 들어간다는 기본적인 encoder decoder의 pipeline을 설명하는 듯 했음.
    - 아래는 official하게 공개된 코드에서 해당되는 부분을 발췌해 온 것
    
    ```python
    class EncoderDecoder(nn.Module):
        def __init__(self, encoder, decoder, src_dense, trg_dense, generator, DEVICE):
            super(EncoderDecoder, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.src_embed = src_dense
            self.trg_embed = trg_dense
            self.prediction_generator = generator
            self.to(DEVICE)
    
        def forward(self, src, trg):
            '''
            src:  (batch_size, N, T_in, F_in)
            trg: (batch, N, T_out, F_out)
            '''
            encoder_output = self.encode(src)  # (batch_size, N, T_in, d_model)
    
            return self.decode(trg, encoder_output)
    
    		def encode(self, src):
    	        '''
    	        src: (batch_size, N, T_in, F_in)
    	        '''
    	        h = self.src_embed(src)
    	        return self.encoder(h)
    	        # return self.encoder(self.src_embed(src))
    	
    		def decode(self, trg, encoder_output):
    	        return self.prediction_generator(self.decoder(self.trg_embed(trg), encoder_output))
    ```
    

## 4.1 Spatial-Temporal Encoder

- ST encoder은 동일한 여러개의 layer으로 구성되어 있는데, 각각의 layer은 **temporal trend aware multi head self attention block**과 **spatial dynamic GCN block**으로 이루어져 있음.
    - **temporal trend aware multi head self attention block은  시간 축에서의 traffic data의 dynamic한 부분을 모델링함**
    - **dynamic GCN block은 spatial한 부분을 capture하고자했음**

### 4.1.1 Temporal Trend-Aware Multi-Head Self-Attention

- Self attention은 attention mechanism 중 하나로, 쿼리 키 벨류가 같은 symbol representation 시퀀스에 포함되어 있는 것을 말함
- Multi head self attention은 가장 광범위하게 채택되는 self attention방법 중 하나로, 각기 다른 subspace에 포함된 representation으로부터 얻은 정보를 jointly attend할 수 있도록 하는 방법임
    - sequence에 포함된 element의 거리와는 무관하게 그들의 correlation을 모델링 할 수 있는 방법으로, global한 receptive field를 효율적으로 갖게 함
- 가장 기본적인 multi head self attention 방법은 scaled dot product attention임.
- 기본적인 transformer 구조에 대해서는 다른 게시글로 정리할 예정
- **Multi head self attention 구조의 초기 버전은 discrete한 token들을 대상으로 구성되어 있음.**
    - 이러한 문제는 아래 사진과 같이, A와 B가 동일한 값을 가지기 때문에 traditional한 self attention을 사용할 경우 mis match가 발생할 수 있다는 문제와 연결됨
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/978b8e6c-99a3-4c68-9b93-d0fc71eb5b1e){: width="50%" height="50%"}{: .center}
    
    - 따라서, 본 연구에서는 continuous 한 data에도 사용될 수 있도록 변형된 temporal trend aware multi head self attention mechanism을 사용함.
        
        ```python
        class SublayerConnection(nn.Module):
            '''
            A residual connection followed by a layer norm
            '''
            def __init__(self, size, dropout, residual_connection, use_LayerNorm):
                super(SublayerConnection, self).__init__()
                self.residual_connection = residual_connection
                self.use_LayerNorm = use_LayerNorm
                self.dropout = nn.Dropout(dropout)
                if self.use_LayerNorm:
                    self.norm = nn.LayerNorm(size)
        
            def forward(self, x, sublayer):
                '''
                :param x: (batch, N, T, d_model)
                :param sublayer: nn.Module
                :return: (batch, N, T, d_model)
                '''
                if self.residual_connection and self.use_LayerNorm:
                    return x + self.dropout(sublayer(self.norm(x)))
                ...
        ```
        
        - SublayerConnection Class는 x와 sublayer을 input으로 받고, 정규화된 x를 sublayer에 전달한 후 dropout시켜준 것을 원본  x와 더해서 return해주는 함수
        
        ```python
        class EncoderLayer(nn.Module):
            def __init__(self, size, self_attn, gcn, dropout, residual_connection=True, use_LayerNorm=True):
                super(EncoderLayer, self).__init__()
                self.residual_connection = residual_connection
                self.use_LayerNorm = use_LayerNorm
                self.self_attn = self_attn
                self.feed_forward_gcn = gcn
                if residual_connection or use_LayerNorm:
                    self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 2)
                self.size = size
        
            def forward(self, x):
                '''
                :param x: src: (batch_size, N, T_in, F_in)
                :return: (batch_size, N, T_in, F_in)
                '''
                if self.residual_connection or self.use_LayerNorm:
                    x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True))
                    return self.sublayer[1](x, self.feed_forward_gcn)
                ......
        ```
        
        - Encoderlayer에서는 sublayer의 0번째 layer에는 x와 self_attn 을 전달하고 return 시 feed_forward_gcn을 sublayer으로 적용함.

### 4.1.2 Spatial Dynamic Graph Convolution

- spatial 한 차원에서의 dynamic을 capture하기 위해서 DGCN(Dynmaic Graph Convolution Net)을 제안함.
- GCN에 대한 간략할 설명(생략해도 좋음)
    - GCN은 structured data에 대한 convolution operation으로 graph에 있는 숨은 비정형적인 패턴을 capturing할 수 있는 network임
    - GCN의 일반적인 idea는 node 에 있는 정보를 교환함으로서 얻을 수 있는 node representation을 학습하고자 했던 것
        - 이웃 representation을 aggregate해서 intermediate representation을 형성하고 aggregated된 representation을 linear projection과 비선형 활성화 함수를 통해 transofrm 시킴.
            - **aggregated한다는건 값을 더해준다기보다는, 합쳐준다는 개념이 강해서 hidden state을 adj matrix과 multiply시켜주는 것으로 이해할 수 있음.**
        - 아래 식에서Z는 node representation을 말하고 **W는 projection matrix**을 의미함.
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/5fc81530-3de6-426e-b60d-366f6c50c844){: width="50%" height="50%"}{: .center}
        
        - A는 node간의 interaction relationship을 represent하고, A물결은 graph 의 adj matrix을 의미한다고 함.
            - A물결이 일반적으로 생각하는 adj matrix인것 같고 거기에 D(모든 j에 대한 A행렬을 더한..? 행렬)이 추가된게 A인듯
                - code에 구현된걸 몾찾겠음
    - 이러한 GCN은 time에 따라서 불변하게 되고 graph에 따른 weight matrix A또한 불변할 것.
        - 하지만, traffic network의 상관관계는 시간에 따라 변하게 되고, 그냥 GCN을 traffic network으로 사용하게 된다면 dynamic한 정보를 포착하는 데에 실패하게 될 것.
- **변화 POINT**
    - **node간의 spatial한 correlation을 동적으로 계산할 수 있도록 self attention을 사용함.**
    - 본 연구에서는 (temporal trend aware multi head atttention의 결과와 같은) node representation이 input으로 들어 올 때, Spatial한 상관관계를 나타내는 Weight matrix St는 아래와 같이 계산됨
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/94ac9c3c-3689-4972-a311-15f2c913be18){: width="50%" height="50%"}{: .center}
        
        - St행렬에서의 Sij는 노드 i와 노드 j간의 correaltion 정도를 나타냄
        - **t시점에서의 Z이니까 시간별로 St는 달라지는것이라고 볼 수 있고, attention을 사용했기 때문에 공간적인 상관관계를 반영해주었다고 주장하는 것 같음.**
        - **St정보를 Weight Matrix A에 반영해줌**
            
            ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/bfb49e77-fb0a-4734-9552-09de31c7e95b){: width="50%" height="50%"}{: .center}
            
    - 본 연구에서 제안된 dynamic graph convolution block은 **변할 수 있는 correlation matrix**에 기반하여 이웃 정보를 aggregate한다고 볼 수 있음.

## 4.2 Spatial-Temporal Decoder

- spatial-temporal decoder은 output sequence를 auto-regressive한 방식을 사용해서 생성함
    - 미래의 subsequence 정보를 사용하는 것을 방지하기 위해서 maksing이 사용됨
- spatial-temporal decoder은 L개의 동일한 decoder layer으로 구성되어 있으며, 각각의 decoder layer은 **2개의 temporal trend-aware multi head attention block**과 **spatial dynamic GCN block**으로 이루어져 있음.
    - 첫번째 temporal trend-aware multi head attention block은 **decoder sequence의 correlation을 capture함.**
        - 미래 정보를 masking하기 위해서 1D conv의 query 와 key부분은 causal convolution으로 대체되었음
            
            ```python
            #class DecoderLayer(nn.Module)의 forward 부분
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False))  # output: (batch, N, T', d_model)
            ```
            
            - 여기서 사용된 self.self_attn은 MultiHeadAttentionAwareTemporalContex_qc_kc(nn.Module) 클래스인데, key query모두 causal 으로 설정되어 있음
    - 두번째 temporal trend-aware multi head attention block에서는 **decoder 의 query sequence와 encoder의 key output sequence의 상관관계를 capture함.**
        - query에만 causal convolution을 적용하고, key에는 일반 convolution을 적용함
            
            ```python
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, query_multi_segment=False, key_multi_segment=True))
            #self.src_attn = attn_st임
            ```
            

## 4.3 Handling Periodicity and Position Embedding

### 4.3.1 Handling Periodicity

- 본 section에서는 global한 주기와 local한 주기에 대해서 다뤄볼 것
    - global 주기는 인간이 정기적으로 하는 행위들을 의미함.
        - ex. 오전 9시에 업무 시작, 오후 6시에 퇴근
    - local한 주기는 종종 일어나는 주기성임
        - ex. 비로 인한 교통 체증
- global/local한 주기를 모두 고려하기 위해서 과거 데이터 외에 추가적인 데이터를 사용함
    - Global Periodic Tensor
        - 과거 w개의 주에서 같은 날을 들고옴
            - ex. 만약 이번주 화요일 오전 9시~오전 11시를 예측하고자 할때 저번주, 저저번주, 저저저번주 화요일의 오전 9시~오전 11시 데이터를 가지고 오는 것
    - Local Periodic Tensor
        - 과거 d개의 날을 들고옴
            - 만약 이번주 화요일 오전 9시~오전 11시를 예측하고자 할때 이번주 월요일, 저번주 일요일의 오전 9시~오전 11시 데이터를 가지고 오는 것
- global/local 정보를 담은 각각의 tensor을 과거 tensor X에 concat시켜서 모델에 input해줌

### 4.3.2 Temporal Position Embedding

- 본 연구에서 제안된 temporal trend-aware 모듈에서는 self attention mechanism을 통해서 dynamic이 모델링됨.
    - attention이 input과 target간의 dependency를 가중합 함수를 통해서 형성하기 때문에, attention mechanism은 시퀀스 속의 symbol순서에 완전히 무관함
    - 하지만 time series modeling task에서의 순서 정보는 중요하기 때문에, 명시적으로 model에게 순서 정보를 주는 것이 모델의 성능을 높일 수 있음.
- X(0)을 position embedding
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