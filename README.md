# Efficient Traffic Prediction through Spatio-Temporal Distillation #
This is the implementation of Efficient Traffic Prediction through Spatio-Temporal Distillation:


## Requirements ##
Pytorch = 1.12.1, python = 3.9.13

## Data ##
PeMSD4, PeMSD8, PeMSD3, PeMSD7, PeMSBay.
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
