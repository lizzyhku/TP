# Efficient Traffic Prediction through Spatio-Temporal Distillation #
This is the implementation of Efficient Traffic Prediction through Spatio-Temporal Distillation:


## Requirements ##
Pytorch = 1.12.1, python = 3.9.13

## Data ##
PeMSD4, PeMSD8, PeMSD3, PeMSD7, PeMSBay.
Because of large data size, we upload the google drive and the link is https://drive.google.com/drive/folders/146BOfs03ljP1OrWnVzdoleuywzCTpS69?usp=sharing.
The historical traffic flows are used from the previous 12 time steps (1 hour) to predict the traffic flows for the next 12 time steps (1 hour). 
Three widely used metrics, namely Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Root Mean Squared Error (RMSE) are adopted to evaluate the accuracy of different traffic prediction models.
## Hyperparameters ##
We use historical 12 time steps' traffic flows (1 hour) to predict next 12 time steps' traffic flows (1 hour). For hyperparameter settings, we set the batch size as 32 for PeMSD4, PeMSD8 and PeMSD3, which could get best performance. We set batch size as 64 for PeMSD7, which achieves the best performance. Through multiple experiments, we set the weights as 10 and 1 for KL divergence and spatial-temporal contrastive module. Besides, when the number of MLP layer is set as 3, LightST achieves the best results.

## How to Run the Code
    python Mains.py --data PeMSD4   
    python Mains.py --data PeMSD8
    python Mains.py --data PeMSD3
    python Mains.py --data PeMSD7
    python Mains.py --data PeMSBAY
Then you can run the code following the order.


If our paper is beneficial for your research, you can cite us as follows:
    
    @inproceedings{zhang2025efficient,
      title={Efficient traffic prediction through spatio-temporal distillation},
      author={Zhang, Qianru and Gao, Xinyi and Wang, Haixin and Yiu, Siu Ming and Yin, Hongzhi},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={39},
      number={1},
      pages={1093--1101},
      year={2025}
}



We also appreciate the help of MTGNN github (https://github.com/nnzhan/MTGNN).
If MTGNN benefits to your research, please cite the MTGNN as follows:

    @inproceedings{wu2020connecting,
      title={Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks},
      author={Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Chang, Xiaojun and Zhang, Chengqi},
      booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
      year={2020}
    }
