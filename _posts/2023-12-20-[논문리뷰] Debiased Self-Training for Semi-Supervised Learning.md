---
layout: post
categories: 논문리뷰
tags: semi-supervised, self-training
comments: true
math: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] Debiased Self-Training for Semi-Supervised Learning"
---
출처
> Chen, Baixu, et al. "Debiased self-training for semi-supervised learning." Advances in Neural Information Processing Systems 35 (2022): 32424-32437.


Copyright of figures and other materials in the paper belongs to original authors.



# Abstract

- self-training is **well-believed to be unreliable** and often leads to training **instability**
    - 이러한 bias를 줄이기 위해서 DST(debiased self-training)을 제안하고자 함.
        - pseudo label들의 generation과 utilization은 직접적인 오류 누적을 피하기 위해서 두개의 independent classifider을 사용해서 진행됨.
        - 최악의 self-training bias를 예측함
            - 여기서 말하는 최악의 상황이란, 수도 라벨링 함수가 labeled sample에 대해서 정확하지만, unlabeled sample에 대해서는 아주 많은.. 오류를 뱉을때를 말함
        - representation을 adversarially optimize해서 수도 라벨들의 퀄리티를 높이고자 함

# 1 Introduction

- self-training is an effective approach to deal with the lack of labeled data.
    - Although self-training has achieved great advances in benchmark datasets**, they still exhibit large training instability** and extreme **performance** **imbalance** across **classes**.
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/ddc90149-9b6f-48d8-8104-97a7d8ba4bf9){: width="30%" height="30%"}{: .center}
        
        - see FixMatch
    - Besides, although FixMatch improves the average accuracy, **it also leads to the Matthew effect**, i.e., the accuracy of well-behaved categories is further increased while that of **poorly-behaved ones is decreased to nearly zero**
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/da2eb029-c792-4bed-b268-cb523a06210e){: width="50%" height="50%"}{: .center}
        
- The above findings are caused by the bias between the **pseudo labeling function** with the **unknown target labeling function.**
- 저자는 bias issuue들을 두가지로 나누어서 생각했음
    - Data bias
        - semi-supervised learning에 내제적으로 존재하는 bias
    - Training bias
        - 부정확한 수도 라벨을 통한 self-training을 해서 증가된 bias
- 이러한 관점에서 Debiased self-training (DST) 를 제안함.
- 특히 training bias를 줄이기 위해서 classifier head는 clean labeled sample들로만 학습하고, unreliable 수도 labeled sample으로는 학습하지 않음 .
- 더 나아가서 data bias를 줄이고자 했는데 이거는 직접적으로 계산되기 힘듦. 따라서, training bias의 최악의 case를 추정함. 이후 최악의 경우 bias를 감소해서 수도 레이블의 퀄리티를 향상시키기 위해서 representation을 obtimize함.

# 2 Related work

- pass

# 3 Analysis of Bias in self-training

- 본 섹션에서는 self-training의 bias가 어디서부터 도출되는지에대해서 분석해보고자 했음
- P가 input space X의 distribution이라고 가정하자. K class들에 대한 분류를 할때, P^k를 class-conditional distrbution of x conditioned on ground truth f*(x)=k라고 하자. 수도라벨러 f_pl이 n labeled samples P_n으롱인해서 획득된다고 가정함. M(f_pl)은 잘 못 생성된 수도라벨 샘플들을 의미함.
- self training 속의 bias를 deviation between the learned decision hyperplanes and the true decision hyperplanes에 대응되며, 이는 모든 클래스에서 잘못된 수도 레이블로 표시된 샘플의 수로 측정될 수 있음.
- 다양한 training condition아래에서 실험을 진행한 결과, nontrivial finding들을 발견함
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/3db564e9-add4-43b1-9ba9-286f4af1a08b){: width="50%" height="50%"}{: .center}
    
    - The sampling of labeled data will largely influence the self-training bias(Figure 1을 보면 알수 있음)
    - The pre-trained representations also affect the self-training bias.
        - Figure 2 shows that different pre-trained representations lead to different category bias, even if the pre-trained dataset and the downstream labeled dataset are both identical.
            - 각기 다른 pre-trained model으로부터 학습된 representatin들은 data의 다른 aspect에 집중한다는 것이 이유가 될 수 있음.
    - Training with pseudo labels aggressively in turn enlarges the self-training bias on some categories(Figure 3을 보면 알 수 있음)
        - 특정 카테고리에 대한 performance gap이 after training with pseudo label, 변화했다는 것을 알 수 있음
- Based on the above observations, we divide the bias caused by self-training into two categories
    - Data bias
        - figure 4의 blue area,
            
            ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/a013b28b-80d6-4539-9b25-c5a82d491d9d){: width="20%" height="20%"}{: .center}
            
        - conditional distrbution of x conditioned on **ground truth f*(x)**
            - 즉, bias가 없는 상황
            - 수도라벨러 f_pl
    - Training bias
        - figure 4의 yellow area
            
            ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/88f7339c-2e78-4312-af64-d9f1b3e0a3ec){: width="20%" height="20%"}{: .center}
            
            - semi를 사용했을때의 상황을 나타냄.
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/8e39536a-2b08-4944-8d80-b5ee5f33395a){: width="50%" height="50%"}{: .center}
        
    - 본 논문에서는 figure4의 빨간색 area 에 해당되는 방법을 제안하고자 함.

# 4 Debiased Self-training

- The standard cross-entropy loss on weakly augmented labeled examples
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/0096decc-d0c3-480d-9f74-d26ac80240b9){: width="50%" height="50%"}{: .center}
    
    - ψ =feature generator
    - h = task-spacific head
    - alpha = weak augmentation function
- Since there are **few labeled samples**, the feature generator and the task-specific head will easily **over-fit**, and **typical SSL method**s use these **pseudo labels** on plenty of unlabeled data to decrease the generalization error.
- Fixmach는 일정 한계점을 통해서 unreliable pseudo label들을 filtering 해줌
- Fixmatch에는 두가지 이슈가 존재함
    1. 수도 라벨이 같은 head를 통해서 생성되고 적용됨 → 이는 training bias로 이어질 수 있음
    2. 엄청나게 적은 labeled sample을통해서 학습 될 경우, confidence threshold mechanism이 소용없을 수 있음. 
- 이를 해결하기 위한 방안을 section 4.1과 4.2에서 제안.

## 4.1 Generate and utilize pseudo labels independently

- Fixmatch의 training bias는 스스로 생성한 수도 라벨을 스스로 학습한다는 것에 있음. 이러한 문제를 해소하기 위해, 아래 figure 5(b), (c)와 같은 방법이 사용될 수 있음.
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/b901ded9-b8cd-4ea2-af18-7370648b18b6){: width="50%" height="50%"}{: .center}
    
    - 하지만 (b)(c)방법에서 모두 수도 라벨을 생성하는 teacher model과 수도 라벨을 활용하는 student model에 강한 relationship이 형성되고, 이에 따라 training bias는 여전히 클 수 밖에 없음.
- 이러한 training bias를 줄이기 위해서, task-specific head를 사용함
    - L : labeled dataset
    - U : unlabeled dataset
    - 적은 labeled sample에 overfitting되는 것을 방지하기 위해서 수도 label을 사용하기는 하지만 더 좋은 representation을 생성하기 위해서 사용함.
        - figure5(d)에서 나와있듯, pseduo head h_pseduo 를 제안하는데 이는 feature generator ψ에 연결되어서 U에 있는 수도 라벨들로만 optimized됨.
        - 따라서 training objective는 아래와 같음.
            
            ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/b4fe2e5b-5dee-4186-9392-e65bfdcedc0d){: width="50%" height="50%"}{: .center}
            
- 수도 라벨은 h으로부터 생성되는데, 독립적인 h_psudo에서 utilized되는 방식임.
- h와h_pseudo가 같은 backbone network으로부터 feature을 받아온다고 해도, 이들의 파라미터가 독립적이기 때문에 pseudo head를 잘못된 수도 라벨으로 학습하는 것은 head h 에 직접적으로 오류를 축적시키기는 않을 것임.
    - pseudo head는 feature generator 의 backpropagation에 responsible하고 inference에서는 사용되지 않는 형식임

## 4.2 Reduce generation of erroneous pseudo labels

- figure6(a)에서 확인할 수 있듯, data bias으로 인해서 각 class의 labeled sample들이 decision hyperplane들로부터의 거리가 각각 다를 수 있음.
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/b00fe485-28d5-4f0e-b9ee-9280169aca7e){: width="50%" height="50%"}{: .center}
    
    - 이는 학습된 hyperplane과 실제 decision hyperplane간의 간극으로 이어질 수 있는데 특히 적은 라벨을 가지고 있을 때 이런 현상이 자주 발생함.
    - 결과적으로 수도 라벨링은 이렇게 **biased decision hyperplane들에 가까운 points들에** **부정확한 수도 라벨**을 할 가능성이 높음.
- 우리가 U를 위한 라벨이 없기 때문에, data bias를 직접적으로 측정하고 그걸 없애는 방법은 없음. 하지만 data bias와 training bias는 어느정도의 상관관계를 보임.
    - section 4.1에서와 같이 task-specific head h는 clean labeled data에 optimized되었음. 이때 부정확한 수도 라벨으로 optimized될 시, learned hyperplane을 더 편향된 방향으로 밀 것이고, training bias는 커질 것임.
    - 따라서, training bias는 data bias의 축적된 결과로 여겨질 수 있음
    - Specifically, the worst training bias corresponds to the worst possible head h 0 learned by pseudo labeling, such that **h 0 predicts correctly on all the labeled samples L** **while making as many mistakes as possible on unlabeled data U** ,where the mistakes of h 0 on unlabeled data are estimated by its discrepancy with the current pseudo labeling function fb.
        - 이 케이스는 figure6(b)임.
- 아래 식은 worst-case of task-specific head h를 찾는 것을 목적으로 함.
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/e25f0df6-1308-4ce3-bfee-8bbdd1c4e881){: width="50%" height="50%"}{: .center}
    
    - Note that Equation 6 measures the degree of data bias, which **depends on the feature representations generated by ψ**, thus we can adversarially optimize feature generator ψ to indirectly decrease the data bias,
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/cc95320b-f77d-42c0-b487-88f3d72c5a6e){: width="50%" height="50%"}{: .center}
        
    - 참고) 기존Fixmach의 loss 함수
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/299d743e-d086-4d9d-9e74-ac8ab242d0fb){: width="50%" height="50%"}{: .center}
        
        - h : the task-specific head.
        - ψ : the feature generator
        - f : pseudo labeling function
        - A : the strong augmentation function.
    - overall loss
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/7aeb1fb5-4455-4768-831f-8de95308fc66){: width="60%" height="60%"}{: .center}
        
        - worst possible head h`에 대해서는 cross-entropy loss를 최대화하고,
        - feature generator, task-specific-head, h_pseudo에 대한 loss는 최소화함.
        

![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/38342850-8f76-40d2-8b4f-e1208989875e){: width="50%" height="50%"}{: .center}

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