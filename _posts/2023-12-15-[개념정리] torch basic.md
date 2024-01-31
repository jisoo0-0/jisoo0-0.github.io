---
layout: post
categories: 개념정리
tags: torch
comments: true
math: true
disqus: 'https-jisoo0-0-github-io' 
title: "[개념정리] torch 초보용 개념정리"
---

# 0. 용어

- **정답 클래스에 대한 확률 수치를 가능도(likelihood)라고 함**

# 1. 모델 생성

## 1-1. 파라미터 설정

### 1-1-1. Optimizer 팁

- 학습 시 정확도가 갑자기 튀는 경향을 보인다면, ADAM에서 SGD로 변경해보는 것도 좋음

### 1-1-2. 파라미터 개수 계산

- code
    
    ```python
    numel_list = [p.numel() for p in connected_model.parameters() if p.requires_grad==True]
    sum(numel_list)
    ```
    

### 1-1-3. 드롭아웃

- 훈련을 반복할 때마다 신경망의 뉴런을 랜덤하게 0으로 설정해줌

# 2. 모델 학습

## 2-1. model.train()

### 2-1-1.model.eval()으로 인한 model.train()의 위치 선정

- 아래 코드와 같이 train 구간 속 valid를 평가하는 함수가 있다면, model.train을 for문 속에 꼭 포함시켜 주어야 함.

```python
for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_pred=[]
    train_y=[]
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        x = torch.tensor(batch[0], dtype=torch.float32, device=device)
        y = torch.tensor(batch[1], dtype=torch.long, device=device)
        with torch.cuda.amp.autocast():
            pred = model(x)
        loss = criterion(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()/len(train_loader)
        train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
        train_y += y.detach().cpu().numpy().tolist()
    scheduler.step()
        
    
    train_f1 = score_function(train_y, train_pred)
    train_accuracy = accuracy_function(train_y, train_pred)
    valid()
    print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}, accuracy : {train_accuracy:.5f}')
    print(f'epoch : {epoch+1}/{epochs}')
```

## 2-2. nn 모듈

### 2-2-1. kaiming_normal_사용법

- 코드를 살펴보면, **torch.nn.init** 모듈에 정의되어 있는 *Kaiming initializer* 함수를 이용해 weight를 초기화 하고 있는 것을 볼 수 있다. Bias 또한 적절한 방식으로 값을 초기화 하고 있다.
- 위의 코드에서 이해할 수 있듯이, 특별한 방식을 쓰고 싶은 것이 아니라면 기본적으로는 **nn.init**에 정의되어 있는 **초기화 함수**를 사용해 모델 컴포넌트의 파라미터를 임의로 초기화 할 수도 있다. 대부분의 함수는 인자로 **weight 텐서**를 입력받는다. (위에서 등장한 Parameter 클래스는 torch.Tensor의 subclass이다)
- 출처 - [https://jh-bk.tistory.com/10](https://jh-bk.tistory.com/10)

### 2-2-2. nn.Sequential

- 한 블럭의 출력을 다른 블럭의 입력으로 사용할 수 있음

# 3. 모델 저장

## 3-1. State dict으로 저장

### 3-1-1. Train시 epoch 돌때마다 저장해주기 꼭..

```python
torch.save(model.state_dict(), f"model/{epoch}_model.pt")
```

# 4. torch tensor 연산

## 4.1 차원 변경

- 
    - 텐서의 복사본을 만들지는 않음
- view vs reshape
    - 공통점
    
    ```python
    daily_bikes = bikes.view(-1, 24, bikes.shape[1]) #24, bikes.shape[1]을 기준으로 0번째차원의 수를 맞춰줌
    ```
    
    - 차이점
        - view는 연속된 공간에 있는 텐서에만 적용 가능

## 4.2 학습 유무

- **requires_grad = True를 통해서 해당 텐서가 학습가능한 텐서인지 아닌지를 판단**
    
    ```python
    params = torch.tensor([1.0, 0.0], requires_grad=True)
    ```
    
    - link : [https://study-grow.tistory.com/entry/pytorch-requires-grad-확인](https://study-grow.tistory.com/entry/pytorch-requires-grad-%ED%99%95%EC%9D%B8)
    - 해당 인자는 params에 가해지는 연산의 결과로 만들어지는 모든 텐서를 이은 전체 트리를 기억하도록 파이토치에게 요청하고 있음
    - **params를 조상으로 두는 모든 텐서는 params로부터 해당 텐서가 만들어지기까지 그 사이에 있는 모든 함수에 접근할 권한을 가짐**
    - **만약 이 함수가 미분 가능하다면, 미분값은 params 텐서의 grad 속성으로 자동 기록됨**
    - 텐서의 수가 몇개이든, 함수 합성이 얼마나 다양하게 이루어지는지와는 별개로 requires_grad를 True로 설정 가능함. 이런 경우 파이토치는 연쇄적으로 연결된 함수들을 거쳐서 손실에 대한 미분을 계산하고 그 값을 텐서의 grad속성에 누적함.

### 4.2.1 with torch.set_grad_enabled(is_train):

- autograd를 켤지말지를 제어할 수 있음
- is_train이 True이면,  아래 식에 대입되는 torch에 대해서 torch.requires_grad가 True
    - ex. y = x*2 일 때 y 는 True, x는 영향안받음

### 4.2.2 assert val_loss.requires_grad == False

- 만약 val_loss.requires_grad가 true이면 오류나는거임

### 4.2.3 optimizer = optim.SGD(linear_model.parameters(), lr = 1e-2)

- 이전에 선형 모델 때에는 params인자를 optim의 파라미터로 넣어주어야 했지만, nn.Module이나 어느 하위 모듈에 대해서도 parameters 메소드로 파라미터 리스트를 얻을 수 있음
- parameters 메소드를 호출하면, 모듈의 init 생성자에 정의된 서브 모듈까지 재귀적으로 호출하고, 만난 모든 파라미터 리스트를 담은 리스트를 반환함
- torch.nn에는 유용한 것이 하나 더 있는데, 바로 손실 계산임
    - nn 자체에 이미 일반적인 손실 함수가 들어 있음

### 4.2.4 nn.Sequential

- nn.Sequential 클래스는 nn.Linear, nn.ReLU(활성화 함수) 같은 모듈들을 인수로 받아서 순서대로 정렬해놓고 입력값이 들어모면 순서대로 모듈을 실행해서 결과값을 리턴함

### 4.2.5 선형적으로 결합 ? tanh 등의 활성화 함수?

- 모델은 한개의 입력 피처로부터 n개의 은닉된 피쳐로 펼쳐지고, 결과값을 tanh 활성 함수로 넘겨서 결과로 나온 13개의 숫자를 하나의 출력 피처로 만들기위해 선형적으로 결합함
- 선형적으로 결합하는거는 하나의 숫자로 만들어주기 위한 방안인듯?

### 4.2.6 SHAPE 확인

- [param.shape for param in seq_model.parameters()]

# 5. 이미지 변환

- PIL 이미지를 파이토치 텐서로 변환
    - torchvision.transforms가 필요함
    - transforms의 toTensor은 출력 텐서의 차원 레이아웃을 C*H*W으로 바꿔줌
    - C는 3개의 채널임. 아래 세개의 이미지가 합쳐서 하나의 이미지로 완성됨
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/ca0703b6-0168-4f41-bd9c-e14f6a28507e){: width="20%" height="20%"}{: .center}
        
    - 
- 원본 PIL 이미지는 0~255 범위인 반면 ToTensor을 사용하면 데이터가 채널당 32비트 부동소수점 형태가 되면서 0.0~1.0 사이의 값으로 범위가 줄어든다.

# 6. 컨볼루션 관련

## 6.1 기본

### 6.1.1 conv1d, conv2d, conv3d

- nn.Conv1d : 시계열용
- nn.Conv2d : 이미지용
- nn.Conv3d : 용적 데이터나 동영상용

### 6.1.2 Conv2d와 Pooling , FC 예시

- Conv2d(입력, 출력 ..)
    - 출력으로 채널 수를 맞춰줌
- nn.MaxPool2d(k)
    - 이미지 사이즈 관련한 코드
- FC = fullyconnected layer로 linear함수로 구현가능

## 6.2 CN1

### 6.2.1 CNN 모델의 목적

- 일반적으로 큰 수의 픽셀을 가진 이미지에서 출발해, 정보를 압축해가며 분류 클래스(의 확률 벡터)로 만들어 가는 것
- 모델의 아키텍처
    - 중간에 나타나는 값의 개수가 점점 줄어드는 모습에 우리가 목표하는 바가 반영되어 있음
        - 컨볼루션의 채널 수가 점점 줄어들고, 풀링에서 픽셀 수가 줄어들며 선형 계층에서는 입력

# 7 GPU 사용하기

## 7.1 gpu 훈련

### 7.1.1 [Module.to](http://Module.to) VS Tensor.to

- Module.to
    - 모듈 인스턴스 자체를 수정
- Tensor.to
    - 새 텐서를 반환함
    - 파라미터를 원하는 디바이스로 이동한 후 옵티마이저를 만드는 식의 구현
- 모델도 GPU로 옮겨야 함. 모델이나 입력을 GPU로 옮기지 않으면 오류가남 (gpu와 cpu입력을 섞어 실행하지 않기 때문)
- **파이토치는 가중치를 저장 할 때 어떤 디바이스를 기억해 두었다가 반대로 읽어들일 때도 해당 디바이스를 사용하기 때문에 GPU에 있던 가중치는 나중에 GPU로 복구될 것.**
- 나중에 어떤 디바이스에서 돌릴지 모르니까 CPU로 신경망을 옮겨서 저장하던가 해야함


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