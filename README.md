# TrustTrade:
The code of TrustTrade framework
## How to run TrustTrade
the main function is in the file "Data_purchase_by_iter.py" and ""Data_purchase_corp_and_beans.py". In the function you can see some options in the main function. To run TrustTrade itself, you could change the budget per round and loop time by changing the function parameter.
The model structure is in the file "run_cifar.py". The forget score calculation code is from the paper"AN EMPIRICAL STUDY OF EXAMPLE FORGETTING DURING DEEP NEURAL NETWORK LEARNING".
Before you run the code, you need to get the datamodel code from here:https://github.com/MadryLab/datamodels-data. You can choose another embedding method if you want.
Our code provides the method of using the CLIP for embedding, in clip_embedding.py
