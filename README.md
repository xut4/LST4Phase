## **MODEL**
* SegPhase
    * **LST4Phase**也在這裡
    * train_cwa.py
    * pred_cwa.py
    * method=
        * LSTCNN:LST4Phase論文final
        * LST:二代模型 LST為分支
        * LFB:初版模型 Learnable Filter Bank
        * ST:S-Transform
        * WL:小波
        * TF:
        * STFT:STFT
        * *else*:SegPhase原版
    * model
        * model_sstnet.py
            * ModelLST
            * **ModelLSTCNNViT**
        * model_spec.py
            * ModelTF
            * ModelWavelet
            * ModelST
        * model_learnableFilter.py
            * ModelLFB
            * sinc_branch.py(邏輯可以參考SincNet論文)
        * model_BandPass.py
            * ModelBP
            * 類似於LFB變成固定放幾個濾波器 一開始用來作比對的
        * model_str.py
            * SegPhase
## **DATA**
* TW2018
    * 論文中的TW資料集
