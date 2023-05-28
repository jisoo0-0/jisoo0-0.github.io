[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
---
layout: post
categories: 논문리뷰
tags: IO, SensorData
comments: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] MagIO: Magnetic Field Strength Based Indoor- Outdoor Detection with a Commercial Smartphones"
---


##### 논문 및 사진 출처
>Ashraf, Imran, Soojung Hur, and Yongwan Park. "MagIO: Magnetic field strength based indoor-outdoor detection with a commercial smartphone." Micromachines 9.10 (2018): 534.
    
<aside>
💡 Magnetic sensor을 기반으로 Indoor / Outdoor을 구분해 내는 연구를 발견해서 정리해보았음. 그런데 sensor data를 기반으로 한 연구들 대부분 수집된 데이터셋에 따라서 모델의 성능이 많이 달라지기 때문에 참고용으로만 읽어두면 좋을듯.

</aside>
# 0 Abstract

- IO systems work with collaborative technologies including the Global Positioning System (GPS), cellular tower signals, Wi-Fi, Bluetooth and a variety of smartphone sensors.
- **GPS**- and **Wi-Fi-based** systems are **power hungry,** and their **accuracy** is severed by limiting factors like **multipath**, **shadowing**, etc.
    - On the other hand, various **built-in smartphone sensors** can be deployed for environmental sensing. Although these sensors can play a crucial role, **yet they are very less studied.**
- The research first investigates the **feasibility of utilizing magnetic field data alone** for IO detection and then **extracts different features suitable for IO detection** to be used in machine learning-based classifiers to discriminate between indoor and outdoor environments.

→ GPS랑 WIFI 기반의 시스템은 **전력 소모가 크고,** multipath나 shadowing같은 이유로 인해서 **분류 정확도가 높지 않음.** 스마트폰에 있는 built-in sensor들을 사용한 연구들은 다양하게 진행되지 않았기에 **본 연구**에서 **built-in sensor을 사용해서 연구를 진행**해 보고자 함.

# 1  Introduction

- The extensive use of smartphones over the globe incited a new paradigm of services collectively called location-based services (LBS), which rely heavily on the location information
    - LBS are provided through mobile apps depending on the user location, which is calculated through mobile devices and communication networks [3,4].
- Many of the services provided by LBS are context aware. Context is defined as any information that can help in characterizing the current situation of an individual or thing, e.g., name, time, device, environment or location

→ LBS에 대한 중요성 언급

## 1.1 GPS based Indoor /Outdoor detection

- In the open **outdoor** environments, usually, many satellites are **available**, so the positioning is **accurate** enough; sometimes to a few centimeters [8]. **However**, **inside** buildings, the satellite signals are absorbed or attenuated due to blocking ceilings and walls. Hence, the **accuracy is severely degraded.**
- In the situations where the user is **close to windows, however, positioning can be performed,** but in many cases, the **error** is **larger** than the **indoor** building area itself. This degradation in GPS signals is used as the main parameter that decides the user state of being indoors or outdoors for GPS-based IO detection. **When the user enters a building, there is a sufficient drop in the signal’s strength,** which can be used to infer if the user is moving from or to the indoor environment

→ outdoor 환경과 같은 경우 위성이 꽤나 정확하기 때문에 positioning하는 것은 크게 어렵지 않음. 하지만 **indoor환경과 같은 경우 위성 신호가 wall이나 ceiling으로 인해서 약해지는 경우가 있기에 positioning 정확도가 낮아짐**

- **Main challenges of GPS based methods**
    - First, in many cases, the signal degradation is not enough to point out the change
    in user state.
    - The second drawback that is very pivotal regarding user IO state is its high power consumption.

→ GPS를 사용했을 때 주요한 challenge는 **신호의 약화**와 **큰 전력 소모**임

## 1.2 Smartphone sensor based indoor/outdoor detection

- Smartphone sensors are used to apply fixes to the GPS’ limitations, and GPS is used opportunistically to save power.
- The sensors used for IO detection include **Wi-Fi, ambient light sensors, iBeacon, proximity sensors and magnetometers**
    - This sensors are called as MEMS (micro-electromechanical-system)

**→ GPS의 단점인 전력소모를 해소하기 위한 방안으로 sensor data가 사용될 수 있음**

## Contributions

- A feasibility study of using the geomagnetic field (referred to as ’magnetic field’ in the rest of the paper) to detect the user IO state.
- The performance appraisal of machine learning-based techniques to predict IO state with smartphone sensor data alone.
- An ensemble-based classifier to perform IO environment classification using magnetic field data from a smartphone.

→ gemoagnetic field를 사용했다는 점, smartphone sensor data만으로도 IO state을 예측할 수 있다는 점, smartphone으로부터 수집된 자기장 데이터를 사용해서 IO 환경 분류를 해낼 수 있다는 점

# 2 An Insight on the Magnetic Field

- There are a few characteristics of magnetic field measurements that are important regarding their use in the current study.
    - The magnitude of the magnetic field is **very smooth over a restricted area;** **however**, it is **disturbed by the presence of ferromagnetic materials** including…
- Additionally, changing the orientations of the device also **leads to changed magnetic field strength.** The magnetic field is represented by seven features including inclination (I), declination (D), the horizontal magnitude (H), the **total magnitude** (F) and magnetic x, y and z. **In practice, F is used mostly, as x, y and z are employable when the device orientation attitude is fixed.** Therefore, **low discernibility** is another characteristic that restricts the use of magnetic fields [19]

→ 자기장은 철과 시멘트와 같은 ferromagnetic material로 인해서 방해받을 수 있는데 이건 실내 환경에 자주 있을법한 재료이기 때문에 사용이 제한될 수 있음

→ 추가적으로 magnetic field에 포함된 feature는 크기 I,D,H,F가 있음. **그런데 x,y,z는 device가 고정된 상태에서만 유의미한 값을 가지기 때문에 total magnitude인 F가 자주 사용됨.** 따라서 이러한 낮은 식별성이 magnetic field를 사용하는 것을 제한함

# 3 Related Work

- pass

# 4 The Feasibility of Using the Magnetic Field for IO Detection

- **They, however, rely on the variance of the magnetic field, which is not a good measure.** We analyze the variance of various samples from indoor and outdoor environments to corroborate this, and the results are shown in Figure 1.
    ![Figure 1. Magnetic variance for the indoor and outdoor environment. (a) Magnetic variance; (b) Indoor & outdoor environment of given variance.](http://drive.google.com/uc?export=view&id=1K6esdZA8m8wxBR7p4AhALmKmVKv65aY9){: width="80%" height="80%"}{: .center}
    
    Figure 1. Magnetic variance for the indoor and outdoor environment. (a) Magnetic variance; (b) Indoor & outdoor environment of given variance.
    
    - Figure1에 나와있듯, indoor과 outdoor 환경에서 비슷한 분산 형태를 보이는 곳들이 많음. 비슷한 구조는 비슷한 magnetic value를 도출해 낸다고 주장.
        - **→ 개인적으로 비슷한 값이 나왔는지 잘 납득이 안됨**
    - 따라서 magenteic feature을 두개의 그룹으로 나눔
        1. self-sufficient and individually unique for the IO enviornment
        2. not individually characteristic of any environment
        
        → unique값을 가지냐 안가지냐에 따라서 그룹을 나눈것 같음. 해당 값을 기반으로 장소에 매칭이 되느냐 안되느냐를 기준으로 본듯
        
        - 첫번째 그룹에 대한 그림이 Figure2인데, 사분위간이나 제곱 편차의 차이가 indoor/outdoor에 있다는걸 보여주고 싶었던 것 같음
            
            ![Figure 2. Magnetic features for the indoor and outdoor environment. (a) Inter-quartile; (b) Squared average deviation.](http://drive.google.com/uc?export=view&id=1JztpKkWewEbYBWzVHtii0LgghnsQ79Hf){: width="80%" height="80%"}{: .center}
            
            Figure 2. Magnetic features for the indoor and outdoor environment. (a) Inter-quartile; (b) Squared average deviation.
            
- Figure 3 shows the plots for kurtosis, and it shows that it is not a good indicator to identify the indoor and outdoor environment. **However, when plotted against the median for the same data, it becomes more meaningful and definitive**
    
    ![Figure 3. (a) Kurtosis and (b) kurtosis and median for the indoor and outdoor environment.](http://drive.google.com/uc?export=view&id=1KiFFDG-1fofq7sE-ptvtYRQAL454IwAn){: width="80%" height="80%"}{: .center}
    
    Figure 3. (a) Kurtosis and (b) kurtosis and median for the indoor and outdoor environment.
    
    - 시간에 따른 kurtosis 값을 보면 indoor과 outdoor을 구별해내기 어려울 수 있는데, 중앙값만 보면 indoor과 outdoor을 구별해낼 수 있음
- The biggest challenge to use magnetic data for IO detection is its low dimension.
    - magnetic field는 7개의 특성으로 표현될 수 있지만, x,y,z,F에만 magnetic value값이 포함되어 있음. 이전에도 mention되었지만 xyz값은 device가 고정되어 있을시에만 유효한 값임. 따라서 F값만 사용할 수 있기 때문에 아주 제한적임.
    - 이와 더불어 F가 IO detection에 좋은 feature가 아님. indoor과 outdoor을 봤을때 F는 아주 비슷한 경향을 보일 수 있음
    - 따라서, extended feature을 통해서 IO state을 분별해내야함.
        - 본 연구에서 16개의 different spatial feature을 extract할 수 있었고 이를 통해서 indoor과 outdoor을 구별할 수 있었음
        - 아래 표를 통해서 이러한 분석을 진행하였음.
            
            ![Untitled](http://drive.google.com/uc?export=view&id=1Ki1cjtMd3Jnhj3br6tk16BOo6h0j1NJ2){: width="80%" height="80%"}{: .center}
            
        - 많은 연구들이 magnetic data을 통해서 IO task을 진행하였지만, magnetic data만을 가지고 IO task를 진행한 연구는 본 연구가 처음임

# 5 Machine Learning Techniques used for classification

- 본 연구에서는 IO detection을 분류 문제로 정의하고, machine learning 기법을 사용했음
    - Decision Trees, K-Nearest Neighbor, Naive Bayes, Random Forest, Gradient Boosting Machines, Rule Induction, SVM 을 사용함

# 6 Experiment and Results

## 6.1 Experimental Setup

- 파란색으로 줄쳐진 부분은 outdoor data가 수집된 장소, 빨간색은 indoor 장소
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1KWoOLVnxqcEqqi5YrYu0rBjwiM9xjbJN){: width="80%" height="80%"}{: .center}
    

## 6.2 Data Collection

- pass

### 6.3 Results

- 아래 figure과 같이 magnetic intensity는 indoor과 outdoor이 확연한 차이를 보임. 하지만 본 연구에서는 magnetic intensity는 사용되지 않았는데, 그 이유로 indoor과 outdoor의 환경이 비슷하다면 비슷한 값이 나오기 때문이라고 주장했음.
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1KV0gcobJZb5SKCyXxDWbnz9hucLXdz8q){: width="80%" height="80%"}{: .center}
    
    - DT, SVM, GMB, RI, kNN classification은 정확도가 50% 언저리였으며 NB와 RF는 그래도 좋으 ㄴ성능을 보였음.
    - 앞서 언급되었듯 ferromagnetic 재질들은 magnetic field의 강도를 방해하고 anomalies를 산출해 낸다. 주차장에 있는 vehicl들과 같은 경우 이러한 이유로 약 1m거리에 있는 magnetic sensor을 방해하였음. 이러한 방해는 magnetic field 의 sudden variance으로 이어졌으며 classification error을 발생하게 했음.
    - 추가적으로 data에 있는 noise또한 낮은 정확도에 기여하였음.
- 또 다른 이유로는 대교가 concrete으로 이어져 있기 때문에 대교 밑으로 걸어갈 때 magnetic field가 방해받았을것이라고 추측함
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1KPDcMl20zj0cptaUAU7m8iWSdGXvPU4a){: width="80%" height="80%"}{: .center}
    
- 앙상블 학습 기법을 사용해서 정확도를 향상시킬 수 있었음
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1KOy4U6S6sfJ0rTECrNnMydK8Yqt2jdAB){: width="80%" height="80%"}{: .center}
    
    - RI랑 DT를 base learner으로 설정하였고,이들은 NB을 통해서 final classification할 수 있었음.
    - data의 noise를 제거하기 위해서 windowing을 사용해서 이동 평균법을 사용함
- 성능을 높이기 위한 또다른 방법으로 데이터를 늘리는 방안이 있음
    
    → 이거 말이 잘 안되게 설명이 되어 있는데 내가 생각했을 때는 그냥 데이터를 더 많이 수집했다는 말인것 같음
    

### 6.4 Performance Comparison and Energy Consumption

- GPS, Wi-Fi, light sensor 데이터도 수집하였음
- 본 연구에서는 추가적인 sensor 데이터가 individual 하게 사용되는 방안, fusion되어서 사용되는 방안을 각각 확인했음
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1KG8WtK5zzVm9B9Esqm4kSZpkah70xOY1){: width="80%" height="80%"}{: .center}
    
    - Wifi 단독으로 사용하기에는 부족해
    
    ![Untitled](http://drive.google.com/uc?export=view&id=1KFb2Dl-YnkwZJCnuaJT6M_A76TlGU1rB){: width="80%" height="80%"}{: .center}
    
    - 위의 figure는 clearly indicates that a clear threshold on RSSI was not very practical for the given scenario and so for the number of scanned APs.
    - light sensor는 날씨에 따라서 영향을 많이 받음.
        
        ![Untitled](http://drive.google.com/uc?export=view&id=1KCeOr8BaGMNP2OXR4JiMreqheVaKWFZ2){: width="80%" height="80%"}{: .center}
        
    - Power consumptions
        
        ![Untitled](http://drive.google.com/uc?export=view&id=1K7zKHLDonZQkhRyT0L1LlKBhYdqk52g_){: width="80%" height="80%"}{: .center}
        
        - 아마 센서에 따라서 소비되는 전력이 다르기 때문에 GPS나 WIFI 말고 다른 센서를 사용했을때의 강점을 강조하고 싶었던 것 같음

# 7 Discussion

- In spite of the importance of IO (indoor-outdoor) systems, this field is not very well studied. IO systems mainly rely on GPS and cellular signal strength.
    
    → 대부분 GPS를 사용한 IO system이 많음
    
    However, during the last few years, smartphone sensors have been investigated for IO detection tasks. Even though, these systems rely heavily on infrastructure-based technologies, i.e., Wi-Fi, Bluetooth, cellular towers, etc., and do not take advantage of smartphone built-in sensors fully.
    
    → 최근에는 smartphone sensor들을 통한 IO detection 연구가 많기는 하지만, infrastructure based기술에 한정되어 있는 연구가 많고 스마트폰 내부의 built int sensor을 온전히 사용한 연구는 거의 없음.
    
- This research is designed to achieve two objectives: a feasibility study on the use of magnetic field alone for IO detection and the use of machine learning to perform IO detection on magnetic data.
    
    → 본 연구는 2개의 목적을 위해서 설계되었음
    
    1. magnetic field의 feasibility한 연구
    2. magentic data를 사용해서 IO detection을 진행
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
