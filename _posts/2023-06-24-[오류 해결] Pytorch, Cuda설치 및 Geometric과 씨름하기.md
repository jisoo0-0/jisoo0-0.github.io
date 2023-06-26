---
layout: post
categories: 오류해결
tags: Pytorch, Cuda, Geometric, 설치오류
comments: true
disqus: 'https-jisoo0-0-github-io' 
title: "[오류해결] Pytorch, Cuda설치 및 Geometric과 씨름하기"
---

기본 설정은 nvidia drivers는 설치된 상태로 간주함. 
Docker기반이긴 하지만 conda 환경이면 무리없이 적용될 것.
(참고)geometric 배포자들은 local환경보다 가상환경을 추천하고 있음.

==귀찮은 사람들은 그냥 python=3.7||3.8 , cuda 11.7으로 세팅해놓고 아래 배포해놓은 requirements.txt 실행 ㄱ==

1. docker hub에 공개되어 있는 cuda11.7, ubuntu 20.04인  이미지를 다운받음

    ```python
    docker pull nvidia/cuda:11.7.0-base-ubuntu20.04
    ```
<br/>
<br/>

2. container 생성함 

    ```python
    sudo nvidia-docker run -it --gpus=all -v 로컬 주소 입력: 컨테이너 속 주소 입력 --shm-size=사이즈크기G --name=컨테이너 이름 docker.io/nvidia/cuda:11.3.0-base-ubuntu18.04 /bin/bash
    ```
<br/>
<br/>

3. container 내에서 아래 코드 실행(필수 패키지랑.. 아나콘다 설치하는거)
    - **만약 어찌어찌하다가 container 속을 벗어났으면 아래 코드를 통해서 다시 접속해줘야함**
        
        ```python
        docker start 컨테이너 이름
        docker attach 컨테이너 이름
        ```
        
        - 컨테이너 접속이 온전하게 잘 됐으면 아래와 같은 이상한 글씨가 당신을 마주할 것
            
            ![Untitled](http://drive.google.com/uc?export=view&id=1fJ3LH71mAesVGEp9BWf77BJqQ7nIhCoT){: width="80%" height="80%"}{: .center}
            
    - 접속이 온전한 것 같으면 아래 코드 입력
        
        ```python
        apt-get update
        apt-get install sudo git build-essential net-tools openssh-server curl gedit autoconf automake libtool make g++ wget vim -y
        wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
        bash Anaconda3-2019.10-Linux-x86_64.sh
        ```
<br/>
<br/>

4. 아나콘다 주소 설정

    ```python
    vi ~/.bashrc
    #vi editor 접속 되면 pgdn키 계속 눌러서 맨 밑부분에 아래 코드 작성
    export PATH=~/anaconda3/bin:~/anaconda3/condabin:$PATH
    #작성 후 esc키 눌러주고 :wq! 입력후 엔터해서 vi editor에서 빠져나옴
    #마지막으로 아래 코드들을 순차적으로 입력
    source ~/.bashrc
    pip install --upgrade pip
    ```
<br/>
<br/>

5. [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive) 여기서 local 선택 후 마지막줄은 실행 하면 안됨 .. please.. 무지성으로 계속 하다가 한시간 날린사람 ✋
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1pTtcursqWXIl8v5bK9nbSF-Qno7r1gDm){: width="80%" height="80%"}{: .center}
    
    - 막줄 대신에 아래 코드 입력함
        
        ```python
        sudo apt-get -y install cuda-11-7
        ```
        
        - 쿠다 뒤에 11-7은 설치하려고하는 cuda 버전임. 본 post에서는 11.7기준이니까 11-7 입력해준거
    - 이렇게 했는데도 안됐다? 그러면 아래 방법 ㄱ
        
        ![Untitled](http://drive.google.com/uc?export=view&id=16y5XdjBpdekI2jhM1uEPGHw_xJO913Gi){: width="80%" height="80%"}{: .center}
        
        - 이거 할 때도 driver 선택은 체크 풀어줘야지 오류 안남
    - nvcc -V 명령어로 원하는 cuda 버전이 제대로 설치되었는지 확인함
    - 아래 코드 실행해서 conda 가상환경 생성하고 해당 가상환경에 접속함
        
        ```python
        conda create -n 원하는가상환경이름 python=3.7
        conda activate 원하는가상환경이름 
        ```
<br/>
<br/>

6. docker을 사용중이고 그 속에서 conda를 이용중이기 때문에 아래와 같은 프로세스를 통해서 geometric을 설치함
    
    -  그냥 아무 파이썬 파일 만들어서 아래 코드 실행
        
        ```python
        TORCH = "1.13.0" #당연히 이부분은 custom하게 바꿔줘야함. 만약 당신이 1.13.1을 설치한 상황이라면 1.13.1을 입력하길
        CUDA = "cu117" #이부분도 마찬가지
        print(f'pip install torch-scatter -f  https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html')
        print(f'pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html')
        print(f'pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html')
        print(f'pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html')
        print('pip install torch-geometric')
        ```
    

    - 출력 결과로 나온 명령어들을 하나씩 터미널에 입력해주면됨
        - 이거 하나하나 설치하는데에 엄~청 오래걸리니까 가능하다면 sh 파일 만들어서 그거 하나 실행해놓는게 좋음
            - sh 파일 만드는 방법은 그냥.. [원하는sh파일이름.sh](http://원하는sh파일이름.sh) 을 생성한 후에 파일 내부에 아래 코드 작성하면됨
                
                ```python
                pip install torch-scatter -f  https://pytorch-geometric.com/whl/torch-$1.13.0+$cu117.html
                pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-$1.13.0+$cu117.html
                pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-$1.13.0+$cu117.html
                pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-$1.13.0+$cu117.html
                pip install torch-geometric
                ```
                
            - 터미널에서 sh [원하는sh파일이름.sh](http://원하는sh파일이름.sh) 으로 실행시켜주면 완성

        - 이때.. 혹시 모를 불상사를 막기 위해서 현재 자신이 속한 환경이 아까 생성한 conda 가상환경이 맞는지 확인함
            - hoxy.. 나해서.. .. conda 가상환경에 들어와 있다면 터미널에서 명령어를 칠 때 왼쪽에 보이는 부분이 아래와 같을건데…
                
                ![Untitled](http://drive.google.com/uc?export=view&id=1GVgd3_V0JyjWFsZcG_OSAPrgiySj2x5J){: width="80%" height="80%"}{: .center}
                
                - () 맨앞 소괄호 안에 아까 지정해준 conda 가상환경 이름이 나와있으면 ok 이고 base 라고 나와있으면 아직 접속 안된거니까 가상환경을 활성화 시켜줘야함
<br/>
<br/>

7. 다 된줄 알았는데 아래 오류가 발생해서 .. 해결해줌 그냥 torch-scatter, torch_geometirc ..등등이 충돌해서 발생하는 오류라.. 아래 링크에 ==pip list 공유해놓음== 원하는 환경에 저 파일 다운로드 받아서 위치시키고, 터미널에서 "pip install -r requirements.txt" 작성하면 됨. 
다운로드 링크 : https://drive.google.com/file/d/11ES5rFbeGv5QJyCWG1F_XKfcgoiTIRX2/view?usp=sharing 

<br/>

- ImportError: cannot import name 'f1_score' from 'torch_geometric.utils' 
- cannot import name 'container_abcs' from 'torch._six'
- RuntimeError: scatter_add() expected at most 5 argument(s) but received 6 argument(s). Declaration: scatter_add(Tensor src, Tensor index, int dim=-1, Tensor? out=None, int? dim_size=None) -> Tensor
- cannot import name 'segment_csr' from 'torch_scatter'


8. 부록) ..이것들 처리하는데에 마주한 main 오류들ㅎ/..하..ㅋㅋ..ㅋ..

    - ImportError: cannot import name 'container_abcs' from 'torch._six’
    - ImportError: cannot import name 'f1_score' from 'torch_geometric.utils’
    - OSError: /root/anaconda3/envs/ ~..  undefined symbol: ~… TensorImpl22is_strides_like_customENS_12MemoryFormat.
    - 3 errors detected in the compilation of "cuda/spspmm_kernel.cu". error: command '/usr/local/cuda/bin/nvcc' failed with exit status 1
    - ERROR: Could not build wheels for torch-sparse-old, which is required to install pyproject.toml-based projects
    - cuda but it is not installed The following packages have unmet dependencies:
    - cuda W: Target Packages (Packages) is configured multiple times in A경로 and B경로
        
        → 이건 gpu랑 cpu를 지원하는 geometric의 버전이 내 컴에 동시에 설치되어서 발생했던 오류임 즉 두개다 삭제하고 다시깔아야함ㅋㅋ
    <br/>


    - 아래 코드들은 다 ubuntu driver 이랑 cuda랑 충돌되었거나 cuda랑 pytorch랑 충돌되었던 상황들에 마주했던 오류들임. 애초에 저런거 해결한다고 아래 코드들 다 사용해봤었는데 삭제 하나도 안되던 상태였음. **아래 코드로 해결안되면 그냥 docker 컨테이너 삭제하고 맨 위에 1번부터 차근차근 따라해보는거 추천**
        
        ```python
        #사용했던 코드모음
        sudo apt-get -f install
        sudo dpkg --configure -a
        apt-get -f install
        ```
        
        - dpkg: error processing package cuda-11-3 (--configure): dependency problems
        - Errors were encountered while processing:
        - cuda E: Sub-process /usr/bin/dpkg returned an error code (1)
        - Processing triggers for dbus (1.12.2-1ubuntu1.4) ... Errors were encountered while processing:
        - nvcc Sub-process /usr/bin/dpkg returned an error code (1)
        - bsudo: dnf: command not found
        - rmmod: ERROR: could not remove module nvidia_drm: Operation not permitted
        - Failed to initialize NVML: Driver/library version mismatch

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