---
layout: post
categories: 논문리뷰
tags: ContrastiveLearning, Audio, Autoencoder
comments: true
disqus: 'https-jisoo0-0-github-io' 
title: "[논문리뷰] Constrastive Audio-Visual Masked Autoencoder"
---



**논문 및 사진 출처**
>Gong, Yuan, et al. "Contrastive audio-visual masked autoencoder." arXiv preprint arXiv:2210.07839 (2022)..    


# 0 Abstract

- 본 연구에서는 Masked Auto-Encoder(MAE) 모델을 single modality에서 audio-visual multi modality로 확장하고자 했음
- contrastive learning과 masked data modeling을 combine함으로서 joint / coordinated audio-visual representation을 학습하고자 했음

# 1 Introduction

- 음향과 시각 modality들은 다른 특성을 가지고 있지만, 인간은 이를 잘 융합해서 세상을 바라볼 수 있음.
    - 인간과 동일한 동작을 할 수 있는 딥러닝 모델에 대한 연구의 필요성
- 사람을 이용한 audio/video annotating은 비용이 너무 많고, scale하기 쉽지않다
    - scale하기 쉽지않다는게 100프로 와닿지는 않음
        - large scale 데이터를 다루기 어려움
    - web-scale unlabeled video data를 self-supervised manner으로 해결하고자 하는 것에 대한 연구가 중요해짐
- multi-modal representation은 두가지 카테고리로 나뉠 수 있음
    - joint representations
        - unimodal signal을 해당 representation space에 combine하는 것
        - 예를 들어 이미지와 텍스트를 같은 representation space에 통합하여 표현가능한 것
    - coordinated representations
        - unimodal signal을 분리해서 process 하지만 특정한 constraint을 해당 signal에 적용함
        - 예를 들어 이미지와 텍스트를 각각의 representation space에 표현하고, similarity 에 따라 특정 제약을 주는 것
- audio-visual self-supervised learning 연구에서 주된 연구 방향은 2가지가 있음
    - 비디오에 포함된 audio-visual correspondences 를 사용하는 것임
        - coordinated representation을 학습함
    - Masked data modeling 방식이 있음
        - corrupted 된 input/feature을 복원하는 pretext task을 사용해서 의미있는 representation을 학습하고자 하는 것
        - joint representation과 관련 있음
- 본 연구에서는 joint와 coordinated representation을 모두 사용하고자 했음
    - 저자는 joint와 coordinated representation을 사용하는 학습방식은 상호 보완적이라고 주장함
    - Constrative audio-visual learning은 유용한 audio-visual pair 정보를 추출해 내지만, modality-unique information을 discard한다는 단점이 있음
    - AV-MAE 의 reconstruction task 는 fusion을 통해서 input information에 대한 대부분의 representation을 인코딩함. 하지만, audio-visual correspondence objective을 discard함
        
        
        |  | Constrative audio-visual learning | MAE / AV-MAE | CAV-MAE |
        | --- | --- | --- | --- |
        | data  | constrative data | masked data | masked data + constrative data |
        | representation | coordinated | reconstructed task으로 인한 meaningful representation / joint(unimodal signal들을 fuse시켜줌) |  |
        | 장점 | 용한 audio-visual pair 정보를 추출해 낼 수 있음 | input information에 대한 대부분의 representation을 인코딩 |  |
        | 단점 | modality-unique information을 discard | audio-visual correspondence objective을 discard |  |
- audio-visual event classification에 대한 실험 결과
    - constrastive / masked data을 통한 모델과 비교해서 outstanding한 성능을 보임
- 본 연구의 contributions
    - single model MAE를 multi modal MAE로 확장
    - constrative learning과 masked data modeling을 best combine 하는 방법을 제안함
    - constrative 과 masked data modeling이 상호 보완적이라는 것을 증명함

# 2 Constrastive audio-visual masked autoencoder

## 2.1 Preliminaries

### 2.1.1 Audio and image pre-processing and tokenization

- AST와 ViT를 통해서 오디오와 이미지 입력을 각각 전처리 / 토큰화 함
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/3bdeaa41-5f42-44ce-b4cf-d82e0b1d768d)

    
    - model의 fine-tune을 위해서 AudioSet의 10초 video와 VGGSound을 사용함
        - audio는 10 second audio waveform이 128차원의 log mel filterbank sequence으로 변환되었음
            - hamming window는 10ms마다 25ms으로 계산됨 / frequency 와 stride (25 가 window lengths, 10ms 가 strides)
                - 이건 국룰 window lengths와 strides. 16K 에 25ms하면..
            - 결과적으로 1024x128 스펙토그램이 생성되었는데, 본 연구에서는 512 개의 16x16의 패치로 나눴음
                - input
                    
                    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/e7881745-d959-426a-a0a1-bfb06466db05)
                    
        - video는 transformer을 상요해서 처리되었음
            - 연산량을 줄이기 위해서 frame aggregation 전략을 사용함
            - RGB frame을 resize 하고 중앙을 기준으로 224x224 사이즈로 cropping함. 이후 196개의 16x16 패치로 나눔
                - input
                    
                    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/dcf4d28c-cac7-4b95-be8d-b5e4b8cf2efc)
                    

### 2.1.2 The transformer architecture

- Conventional 한 Multi-head transformer 을 사용함

### 2.1.3 Constrative audio-visual learning(CAV)

- self-supervision을 위해서는 video에 있는 audio-visual pair representation이 유용함
- Figure 1.B에 CAV가 나와있음
    - audio-visual pair sample의 N 배치
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/d282242f-1f4b-4b9e-a691-2e3248319bff){: width="30%" height="30%"}{: .center}

        
    - 먼저 audio와 image를 pre-process 하고 tokenize을 해서, audio , visual sequence token {a_i, v_i}를 각 sample i 마다 얻을 수 있게 됨
    - 이후 각 a_i, v_i를 E_A와 E_V에 입력해주고 결과값을 pooling해줌 (여기서 개수 맞춰짐)
    - c_i^a 와 c_i^v에 대해서 constrastive loss 를 적용해줌
        - 이게 왜 constrastive loss 인가?
            
            ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/b0a2c2f3-c466-4812-98e4-d6fc5f538edd){: width="40%" height="40%"}{: .center}
            ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/cc02dcbe-b5ec-4818-b882-4a300f6fce6c){: width="80%" height="80%"}{: .center}
            
            - 내적은 각 vector의 similarity 를 나타내는데, 위의 식에서는 s_{i,i}의 값은 작아지게 s_{i,k}의 값은 커지게 됨. 즉 해당 수식을 loss 함수로 사용하게 된다면, 같은 샘플에 대한 vector간의 값은 낮아지고 다른 vector간의 값은 높아게 됨
            ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/f008c9d1-bcb2-4faa-abb7-b4363f40e365){: width="80%" height="80%"}{: .center}


### 2.1.4 Single modality masked autoencoder(MAE)

- input sample x 는 [x^1, .. x^n]으로 tokenized 될 수 있음.
- MAE는 input x_mask의 일부를 마스킹하고, 마스킹되지 않은 토큰들만 transformer 모델에 입력됨.
- 모델은 MSE loss 를 줄이는 방향으로 masked token을 reconstruct함.
    - 해당 process 에서 모델은 input data의 meaningful representation을 학습하게 됨.
- MAE을 사용할 때의 이점
    - MAE는 prediction target으로 original input을 직접적으로 사용하기 때문에 training pipeline을 simplify함
    - MAE는 unmaksed token만을 사용하고, 이를 high maksing ratio와 combine시켜줌. → 연산량을 줄여줌
    - MAE는 audio-visual modal각각 seperate된 task에서는 좋은 성능을 보였음
## 2.2 Vanila audio-visual masked autoencoder(AV-MAE)

- MAE가 지금까지 audio 와 video 각각의 modality 에 대해서는 적용되어 왔지만, audio-video multi modality learning에서는 사용된적없음.
- 아래 사진에서와 같이, audio / image input을 tokenize(a^1,… a^512) / (v^1…v^196) 하고 768 차원으로 projection 시켜줌.
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/1b7af0ab-dbf1-4d5b-b142-0c68508cdbe4){: width="40%" height="40%"}{: .center}

    
    - projection layer
        - 아마 mask concatenate 쪽인거같고, 관련된 식은 2 3인듯
            
            ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/07260e86-f4b2-47bb-96b4-ecaa7a5022d8){: width="80%" height="80%"}{: .center}

            
            - Where E_a, E_v are the embeddings of the modality type
                
                ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/835251e2-a42d-4e08-8657-6eda2c96c172){: width="80%" height="80%"}{: .center}
                
    - E_j 는 joint encoding
    - Reconstruction loss
        
        ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/447b5f79-cb7c-42a7-9162-3ff195df056c){: width="80%" height="80%"}{: .center}

        

# 3 Code review

## 3.1 data preprocessing.py

### 3.1.1 extract_audio.py

- code
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/1278acb9-bba9-43fc-a060-28ddf93b493d){: width="80%" height="80%"}{: .center}
    
    - input_f → 파일명(주소도 포함)
    - ext_len = 파일명 길이
    - video_id = 비디오 파일명 길이
    - output_f_1 = output될 file의 이름
    - ffmpeg
        - 오디오 변환 툴  사용
        - 샘플링 rate 16000(16kHz)
    - sox
        - 첫번재 채널을 처리하기 위한 툴인것으로 보임
            - remix라는 명령어가 스테레오 파일을 모노 파일로 remix해주는 명령어

### 3.1.2 extract_video_frame.py

- extract_frame
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/63f4fa1e-9aaf-4c08-bb62-2cfe64147c66){: width="80%" height="80%"}{: .center}

    
    - file 개수마다 호출됨
    - cv2.VideoCapture
        - video 처리해주는 모듈 호출
    - vidcap.get(cv2.CAP_PROP_FPS)
        - 초당 프레임 수를 구함
    - total_frame_num
        - 최소 프레임은 초당 프레임 수 * 10
    - total_frame_num/extract_frame_num
        - 말그대로 frame을 frame_num으로 나누면 여러개의 구간이 생기는데, for 문 속에서 현재 몇번째 구간인지(frame 인지)를 나타내줌
    - cv2.set으로 frame 설정 이후 read으로 frame읽어옴
    - cv2_im는 RGB로 변환된 이미지가 담기고, pill_im을 preprocess 작업을 거쳐준 이후 save해줌.
        - preprocess인자는 논문에 나온 것처럼 resize, crop 해주는거

## 3.2 sample data

- sample table(class_labels_indices_as.csv)
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/f0e0099c-e97a-4cee-81b4-b73bb093446f){: width="40%" height="40%"}{: .center}
    
    - display_name : label
    - mid : json파일과 매칭될 수 있는 label(unique key)
- sample_json_as.json
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/34d04377-0166-46ab-8d69-8897e759954e){: width="40%" height="40%"}{: .center}


## 3.3 **create_json_as.py**

- code
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/88631c0e-b657-4557-8bd2-3903526f3b59){: width="80%" height="80%"}{: .center}
    
    - 위의 sample_json_as.json 을 전처리 하는게 아니라, 진짜 metadata를 의미하는것. sample_json_as.json은 create_json_as의 결과물.

## 3.4 Dataloader

### 3.4.1 make_index_dict

- code
    
    ![image](https://github.com/jisoo0-0/jisoo0-0.github.io/assets/130432190/7521f65f-54cf-4988-b57f-c724d6bb1607){: width="40%" height="40%"}{: .center}
    
    - index_lookup
        - unique key와 index 매칭해줌

### 3.4.2 AudiosetDataset

**init**

- code
    
    ```python
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
            """
            Dataset that manages audio recordings
            :param audio_conf: Dictionary containing the audio loading and preprocessing settings
            :param dataset_json_file
            """
            self.datapath = dataset_json_file
            with open(dataset_json_file, 'r') as fp:
                data_json = json.load(fp)
    
            self.data = data_json['data']
            self.data = self.pro_data(self.data) #list를 numpy array로 변환
            print('Dataset has {:d} samples'.format(self.data.shape[0]))
            self.num_samples = self.data.shape[0]
            self.audio_conf = audio_conf
            self.label_smooth = self.audio_conf.get('label_smooth', 0.0) 
            print('Using Label Smoothing: ' + str(self.label_smooth))
            self.melbins = self.audio_conf.get('num_mel_bins') #mel spectrogram의 bin개수 설정
            self.freqm = self.audio_conf.get('freqm', 0) #frequency mask의 최대 길이 설정
            self.timem = self.audio_conf.get('timem', 0) #time mask의 최대 길이 설정
            print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
            self.mixup = self.audio_conf.get('mixup', 0) 
    				#training 동안 얼마나 많은 sample들이 mixup되어야 하는지
    				#maybe.... data augmentation
    
            print('now using mix-up with rate {:f}'.format(self.mixup))
            self.dataset = self.audio_conf.get('dataset')
            print('now process ' + self.dataset)
            # dataset spectrogram mean and std, used to normalize the input
            self.norm_mean = self.audio_conf.get('mean')
            self.norm_std = self.audio_conf.get('std')
            # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
            # set it as True ONLY when you are getting the normalization stats.
            self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
            if self.skip_norm:
                print('now skip normalization (use it ONLY when you are computing the normalization stats).')
            else:
                print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
    
            # if add noise for data augmentation
            self.noise = self.audio_conf.get('noise', False)
            if self.noise == True:
                print('now use noise augmentation')
            else:
                print('not use noise augmentation')
    
            self.index_dict = make_index_dict(label_csv)
            self.label_num = len(self.index_dict)
            print('number of classes is {:d}'.format(self.label_num))
    
            self.target_length = self.audio_conf.get('target_length')
    
            # train or eval
            self.mode = self.audio_conf.get('mode') #train / eval mode를 말함
            print('now in {:s} mode.'.format(self.mode))
    
            # set the frame to use in the eval mode, default value for training is -1 which means random frame
            self.frame_use = self.audio_conf.get('frame_use', -1)
            # by default, 10 frames are used
            self.total_frame = self.audio_conf.get('total_frame', 10)
            print('now use frame {:d} from total {:d} frames'.format(self.frame_use, self.total_frame))
    
            # by default, all models use 224*224, other resolutions are not tested
            self.im_res = self.audio_conf.get('im_res', 224)
            print('now using {:d} * {:d} image input'.format(self.im_res, self.im_res))
            self.preprocess = T.Compose([
                T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
                T.CenterCrop(self.im_res),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.4850, 0.4560, 0.4060],
                    std=[0.2290, 0.2240, 0.2250]
                )])
    
    ```
    

**pro data**

- wav2fbank
    
    ```python
    def _wav2fbank(self, filename, filename2=None, mix_lambda=-1):
            # no mixup
            if filename2 == None:
                waveform, sr = torchaudio.load(filename) #audio load
                waveform = waveform - waveform.mean() #audio 처리에 많이 사용되는 기법
            # mixup
            else: #data 두개를 mix하는데, 길이에 따라 padding 및 cutting작업 진행
                waveform1, sr = torchaudio.load(filename) 
                waveform2, _ = torchaudio.load(filename2)
    
                waveform1 = waveform1 - waveform1.mean()
                waveform2 = waveform2 - waveform2.mean()
    
                if waveform1.shape[1] != waveform2.shape[1]:
                    if waveform1.shape[1] > waveform2.shape[1]:
                        # padding
                        temp_wav = torch.zeros(1, waveform1.shape[1])
                        temp_wav[0, 0:waveform2.shape[1]] = waveform2
                        waveform2 = temp_wav
                    else:
                        # cutting
                        waveform2 = waveform2[0, 0:waveform1.shape[1]]
    
                mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2 #어느정도 비율로 두개의 audio를 섞어줄지를 정해줌
                waveform = mix_waveform - mix_waveform.mean() 
    
            try: # filter bank feature extraction 부분. mel scale 변환으로 마무리. 
                fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
            except:
                fbank = torch.zeros([512, 128]) + 0.01
                print('there is a loading error')
    
            target_length = self.target_length
            n_frames = fbank.shape[0]
    
            p = target_length - n_frames
    
            # cut and pad , 입력길이를 맞춰주기 위한 방법
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)
            elif p < 0:
                fbank = fbank[0:target_length, :]
    
            return fbank
    ```
    

**getitem**

- code
    
    ```python
    def __getitem__(self, index):
            if random.random() < self.mixup:
                datum = self.data[index]
                datum = self.decode_data(datum) #이전에 변환한 numpy array를 dic으로 다시 변환해주는 과정임 
                mix_sample_idx = random.randint(0, self.num_samples-1) 
                mix_datum = self.data[mix_sample_idx]
                mix_datum = self.decode_data(mix_datum)  #이것도 dictionary 형태로 변환해줌. 
                # get the mixed fbank
                mix_lambda = np.random.beta(10, 10) #audio 1과 audio2를 어느정도 비율로 섞어줄건지 random하게 설정 
                try:
                    fbank = self._wav2fbank(datum['wav'], mix_datum['wav'], mix_lambda) 
                except:
                    fbank = torch.zeros([self.target_length, 128]) + 0.01
                    print('there is an error in loading audio')
                try: #전처리된 이미지 들고옴. 전처리는 Resize, centercrop, normalize 적용
                    image = self.get_image(self.randselect_img(datum['video_id'], datum['video_path']), self.randselect_img(mix_datum['video_id'], datum['video_path']), mix_lambda)
                except:
                    image = torch.zeros([3, self.im_res, self.im_res]) + 0.01
                    print('there is an error in loading image')
                label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num) 
                for label_str in datum['labels'].split(','): #labels: "/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"
                    label_indices[int(self.index_dict[label_str])] += mix_lambda * (1.0 - self.label_smooth)
    #self.index_dict을 통해서 해당 라벨의 인덱스를 호출하고 label_indices에 반영해줌.
    #mix_lambda르 곱해주는걸 보면 mixup 할때의 smoothing을 조절해주는 것 같기는 한데 코드랑 논리가 매칭이잘안됨..
                for label_str in mix_datum['labels'].split(','):
                    label_indices[int(self.index_dict[label_str])] += (1.0 - mix_lambda) * (1.0 - self.label_smooth)
                label_indices = torch.FloatTensor(label_indices)
    
            else:
                datum = self.data[index]
                datum = self.decode_data(datum) #딕셔너리 변환
                # label smooth for negative samples, epsilon/label_num
                label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
                try:
                    fbank = self._wav2fbank(datum['wav'], None, 0) 
                except:
                    fbank = torch.zeros([self.target_length, 128]) + 0.01
                    print('there is an error in loading audio')
                try:
                    image = self.get_image(self.randselect_img(datum['video_id'], datum['video_path']), None, 0)
                except:
                    image = torch.zeros([3, self.im_res, self.im_res]) + 0.01
                    print('there is an error in loading image')
                for label_str in datum['labels'].split(','):
                    label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
                label_indices = torch.FloatTensor(label_indices)
    
            # SpecAug, not do for eval set
    				#masking 적용
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm) 
            timem = torchaudio.transforms.TimeMasking(self.timem)
            fbank = torch.transpose(fbank, 0, 1)
            fbank = fbank.unsqueeze(0)
            if self.freqm != 0:
                fbank = freqm(fbank)
            if self.timem != 0:
                fbank = timem(fbank)
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)
    
            # normalize the input for both training and test
            if self.skip_norm == False:
                fbank = (fbank - self.norm_mean) / (self.norm_std)
            # skip normalization the input ONLY when you are trying to get the normalization stats.
            else:
                pass
    
            if self.noise == True:
                fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
                fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0)
    
            # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
            return fbank, image, label_indices
    ```
    

## 3.5 **cav_mae.py**

CAVMAE class

- **initiation**
    
    ```python
    class CAVMAE(nn.Module):
        """ CAV-MAE Model
        """
        def __init__(self, img_size=224, audio_length=1024, patch_size=16, in_chans=3,
                     embed_dim=768, modality_specific_depth=11, num_heads=12,
                     decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                     mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, tr_pos=False):
            super().__init__()
            print('A CAV-MAE Model')
            print('Use norm_pix_loss: ', norm_pix_loss)
            print('Learnable Positional Embedding: ', tr_pos)
    
            # the encoder part
            # overide the timm package
            timm.models.vision_transformer.PatchEmbed = PatchEmbed
            timm.models.vision_transformer.Block = Block
    
            self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim) #audio embedding patch embedding
            self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)#audio embedding patch embedding 
    
            self.patch_embed_a.num_patches = int(audio_length * 128 / 256) #embedding patch 개수 
            print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(self.patch_embed_a.num_patches, self.patch_embed_v.num_patches))
    #self.patch_embed_v.num_patches선언이 누락됨. self.patch_embed_v.num_patches = int(224//16)^2
    
            self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim)) 
            self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
    
            self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
            self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
    #sin-cos embedding안되어있음. 
    
            # audio-branch
            self.blocks_a = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(modality_specific_depth)])
            # visual-branch
            self.blocks_v = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(modality_specific_depth)])
            # unified branch
            self.blocks_u = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(12-modality_specific_depth)])
    
            # independent normalization layer for audio, visual, and audio-visual
            self.norm_a, self.norm_v, self.norm = norm_layer(embed_dim), norm_layer(embed_dim), norm_layer(embed_dim)
    
            # the decoder part
            # Project to lower dimension for the decoder
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
    
            # token used for masking
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
    
            self.decoder_modality_a = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_modality_v = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
    
            self.decoder_pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, decoder_embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
            self.decoder_pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, decoder_embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
    
            self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(decoder_depth)])
    
            self.decoder_norm = norm_layer(decoder_embed_dim)
    
            # project channel is different for two modality, use two projection head
            self.decoder_pred_a = nn.Linear(decoder_embed_dim, patch_size ** 2 * 1, bias=True)  # decoder to patch
            self.decoder_pred_v = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
    
            self.norm_pix_loss = norm_pix_loss
    
            self.initialize_weights()
    
            print('Audio Positional Embedding Shape:', self.pos_embed_a.shape)
            print('Visual Positional Embedding Shape:', self.pos_embed_v.shape)
    ```
    
- **init weight**
    
    ```python
    def initialize_weights(self):
            # initialize (and freeze) pos_embed by sin-cos embedding, opt the cls token, add by myself
    				#positinoal embedding함
            pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
            self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))
    
            pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
            self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))
    
            decoder_pos_embed_a = get_2d_sincos_pos_embed(self.decoder_pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
            self.decoder_pos_embed_a.data.copy_(torch.from_numpy(decoder_pos_embed_a).float().unsqueeze(0))
    
            decoder_pos_embed_v = get_2d_sincos_pos_embed(self.decoder_pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
            self.decoder_pos_embed_v.data.copy_(torch.from_numpy(decoder_pos_embed_v).float().unsqueeze(0))
    
            # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
            w = self.patch_embed_a.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            w = self.patch_embed_v.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    
            # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
            torch.nn.init.normal_(self.modality_a, std=.02)
            torch.nn.init.normal_(self.modality_v, std=.02)
            torch.nn.init.normal_(self.decoder_modality_a, std=.02)
            torch.nn.init.normal_(self.decoder_modality_v, std=.02)
            torch.nn.init.normal_(self.mask_token, std=.02)
    
            # initialize nn.Linear and nn.LayerNorm
            self.apply(self._init_weights)
    
        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                # we use xavier_uniform following official JAX ViT:
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
        def patchify(self, imgs, c, h, w, p=16):
            """
            imgs: (N, 3, H, W)
            x: (N, L, patch_size**2 *3)
            """
            x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
            return x
    
        def unpatchify(self, x, c, h, w, p=16):
            """
            x: (N, L, patch_size**2 *3)
            imgs: (N, 3, H, W)
            """
            assert h * w == x.shape[1]
    
            x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
            x = torch.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
            return imgs
    ```
    
- **get_2d_sincos_pos_emb는 아래 코드를 통해서 연산됨**
    
    ```python
    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)
    
        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    
        emb_sin = np.sin(out) # (M, D/2)
        emb_cos = np.cos(out) # (M, D/2)
    
        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D) #sin / cos 적용된 값들을 concatenation 시켜줌. 
        return emb
    ```
    
- **masking**
    
    ```python
    def random_masking_unstructured(self, x, mask_ratio):
            """
            Perform per-sample random masking by per-sample shuffling.
            Per-sample shuffling is done by argsort random noise.
            x: [N, L, D], sequence
            """
            N, L, D = x.shape  # batch, length, dim
            len_keep = int(L * (1 - mask_ratio))
    
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove 
            ids_restore = torch.argsort(ids_shuffle, dim=1)
    
            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    
            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
    
            return x_masked, mask, ids_restore
    
        def random_masking_structured(self, x, mask_ratio, t=64, f=8, mode='time'):
            """
            Perform per-sample random masking by per-sample shuffling.
            Per-sample shuffling is done by argsort random noise.
            x: [N, L, D], sequence
            """
            N, L, D = x.shape  # batch, length, dim
            len_keep = int(L * (1 - mask_ratio))
    
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            assert L == f * t
            noise = noise.reshape(N, f, t) # the audio patch is in shape [f,t], not [t,f]
            if mode == 'time':
                for i in range(N):
                    mask_t_list = random.sample(range(t), int(t * mask_ratio))
                    for k in mask_t_list:
                        noise[i, :, k] = 1.1  # large value will be removed
            elif mode == 'freq':
                for i in range(N):
                    mask_f_list = random.sample(range(f), int(f * mask_ratio))
                    for k in mask_f_list:
                        noise[i, k, :] = 1.1  # large value will be removed
            elif mode == 'tf':
                for i in range(N):
                    mask_t_list = random.sample(range(t), int(t * mask_ratio * 0.7))
                    for k in mask_t_list:
                        noise[i, :, k] = 1.1  # large value will be removed
                for i in range(N):
                    mask_f_list = random.sample(range(f), int(f * mask_ratio * 0.7))
                    for k in mask_f_list:
                        noise[i, k, :] = 1.1  # large value will be removed
            noise = noise.reshape(N, L)
    
            # sort noise for each sample, only need to manuplate these two ids_shuffle, ids_restore
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)
    
            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    
            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
    
            return x_masked, mask, ids_restore
    ```
    
- **forward encoder**
    
    ```python
    def forward_encoder(self, a, v, mask_ratio_a, mask_ratio_v, mask_mode='unstructured'):
            # embed patches
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a) #768 차원 embedding
            a = a + self.pos_embed_a #positional embedding 더함
            a = a + self.modality_a #modality embedding 더함
    
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v
    
            # by default, we always use unstructured masking
            if mask_mode == 'unstructured':
                a, mask_a, ids_restore_a = self.random_masking_unstructured(a, mask_ratio_a)
            # in ablation study, we tried time/freq/tf masking. mode in ['freq', 'time', 'tf']
            else:
                a, mask_a, ids_restore_a = self.random_masking_structured(a, mask_ratio_a, t=64, f=8, mode=mask_mode)
    
            # visual branch always use unstructured masking
            v, mask_v, ids_restore_v = self.random_masking_unstructured(v, mask_ratio_v)
    
            # audio and visual stream, independent blocks
            for blk in self.blocks_a: #위에서 선언된 audio block에 입력값 넣어줌
                a = blk(a)
    
            for blk in self.blocks_v:
                v = blk(v)
    
            x = torch.cat((a, v), dim=1) #concat해줘서 joint representation 생성
    
            # unified stream, shared blocks_u, but independent normalization layers -> 논문과 동일하게 blocks_u를 통해서 동일한 공간에서 학습되지만 
            for blk in self.blocks_u:  
                x = blk(x)
            x = self.norm(x) #norm layer은 개별적으로 적용
    
            for blk in self.blocks_u:
                ca = blk(a, 'a')
            ca = self.norm_a(ca) #norm layer은 개별적으로 적용
    
            for blk in self.blocks_u:
                cv = blk(v, 'v')
            cv = self.norm_v(cv)#norm layer은 개별적으로 적용
    
            return x, mask_a, ids_restore_a, mask_v, ids_restore_v, ca, cv
    ```
    
- **forward decoder**
    
    ```python
    def forward_decoder(self, x, mask_a, ids_restore_a, mask_v, ids_restore_v):
    
            x = self.decoder_embed(x)
    
            # append mask tokens to sequence
            # mask_tokens_a in shape [B, #a_mask_token, mask_token_dim], get the number of masked samples from mask_a[0], which is the first example of the batch, all samples should have same number of masked tokens
            mask_tokens_a = self.mask_token.repeat(x.shape[0], int(mask_a[0].sum()), 1)
            a_ = torch.cat([x[:, :self.patch_embed_a.num_patches-int(mask_a[0].sum()), :], mask_tokens_a], dim=1)  # no cls token
            a_ = torch.gather(a_, dim=1, index=ids_restore_a.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle , ids_restore_a에 있는 원소들을 indexing해줌. 
    
            # similar for the visual modality
            mask_tokens_v = self.mask_token.repeat(x.shape[0], int(mask_v[0].sum()), 1)
            v_ = torch.cat([x[:, self.patch_embed_a.num_patches-int(mask_a[0].sum()):, :], mask_tokens_v], dim=1)  # no cls token
            v_ = torch.gather(v_, dim=1, index=ids_restore_v.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    
            # concatenate audio and visual tokens
            x = torch.cat([a_, v_], dim=1)
    
            decoder_pos_embed = torch.cat([self.decoder_pos_embed_a, self.decoder_pos_embed_v], dim=1)
            x = x + decoder_pos_embed
    
            # add modality indication tokens
            x[:, 0:self.patch_embed_a.num_patches, :] = x[:, 0:self.patch_embed_a.num_patches, :] + self.decoder_modality_a #modality 입력해줌. 
            x[:, self.patch_embed_a.num_patches:, :] = x[:, self.patch_embed_a.num_patches:, :] + self.decoder_modality_v
    
            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)
    
            # predictor projection
            x_a = self.decoder_pred_a(x[:, :self.patch_embed_a.num_patches, :])#Linear(embed dim-> patch_size**2*1)
            x_v = self.decoder_pred_v(x[:, self.patch_embed_a.num_patches:, :])#Linear(embed dim-> patch_size**2*in_chans)
    
            # return audio and video tokens
            return x_a, x_v
    ```
    
- **forward constrastive**
    
    ```python
    def forward_contrastive(self, audio_rep, video_rep, bidirect_contrast=False): #encoder ouput 기준임
            # calculate nce loss for mean-visual representation and mean-audio representation
    
            audio_rep = torch.nn.functional.normalize(audio_rep, dim=-1) 
            video_rep = torch.nn.functional.normalize(video_rep, dim=-1)
    
            total = torch.mm(audio_rep, torch.transpose(video_rep, 0, 1)) / 0.05 ##audeo representation 
    
            # by default we use single directional
            if bidirect_contrast == False:
                nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
                c_acc = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
                return nce, c_acc
            else:
                nce_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
                nce_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total.t(), dim=0)))
                c_acc_1 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
                c_acc_2 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total.t(), dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
                nce = (nce_1 + nce_2) / 2
                c_acc = (c_acc_1 + c_acc_2) / 2
                return nce, c_acc
    ```
    
- **forward mae loss**

```python
def forward_mae_loss(self, input, pred, mask, modality):#decoder ouput 기준임
        if modality == 'a':
            # for audio, need to adjust the shape
            input = input.unsqueeze(1)
            input = input.transpose(2, 3)
            target = self.patchify(input, 1, int(input.shape[2]/self.patch_embed_a.patch_size[0]), int(input.shape[3]/self.patch_embed_a.patch_size[1]), 16)
        elif modality == 'v':
            target = self.patchify(input, 3, int(input.shape[2]/self.patch_embed_v.patch_size[0]), int(input.shape[3]/self.patch_embed_v.patch_size[1]), 16)

        # patch-wise normalization might minorly improve the classification performance, but will make the model lose inpainting function
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
```

- **모델 forwarding**

```python
def forward(self, audio, imgs, mask_ratio_a=0.75, mask_ratio_v=0.75, mae_loss_weight=1., contrast_loss_weight=0.01, mask_mode='unstructured'):
        # latent is used for reconstruction (mae), latent_c_{a,v} are used for contrastive learning
        latent, mask_a, ids_restore_a, mask_v, ids_restore_v, latent_c_a, latent_c_v = self.forward_encoder(audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode)
        # if mae loss is used
        if mae_loss_weight != 0:
            pred_a, pred_v = self.forward_decoder(latent, mask_a, ids_restore_a, mask_v, ids_restore_v)
            loss_mae_a = self.forward_mae_loss(audio, pred_a, mask_a, 'a')
            loss_mae_v = self.forward_mae_loss(imgs, pred_v, mask_v, 'v')
            loss_mae = mae_loss_weight * (loss_mae_a + loss_mae_v)
        else:
            loss_mae_a, loss_mae_v, loss_mae = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)

        # if contrastive loss is used
        if contrast_loss_weight != 0:
            # note this is single directional
            loss_c, c_acc = self.forward_contrastive(latent_c_a.mean(dim=1), latent_c_v.mean(dim=1))
            loss_c = contrast_loss_weight * loss_c
        else:
            loss_c, c_acc = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)

        loss = loss_mae + loss_c

        return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc
```

- **retrival 시 사용되는 forward_feat**
    
    ```python
    def forward_feat(self, a, v): #encoder과 비슷한 로직이지만 unified branch가존재하지않음. 
            # embed patches
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a
    
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v
    
            # the modality-specific stream
            for blk in self.blocks_a:
                a = blk(a)
    
            for blk in self.blocks_v:
                v = blk(v)
    
            # use modality specific normalization,
            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)
    
            for blk in self.blocks_u:
                v = blk(v, 'v')
            v = self.norm_v(v)
    
            return a, v
    ```



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