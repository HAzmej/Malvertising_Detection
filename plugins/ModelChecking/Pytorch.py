def model_pytroch(X_train,y_train,X_test,y_test,BATCH_SIZE,HIDDEN_UNITS):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device

    x_tensor_train=torch.from_numpy(X_train.values).type(torch.float)
    x_tensor_test=torch.from_numpy(X_test.values).type(torch.float)
    y_tensor_train = torch.from_numpy(y_train.values).type(torch.float).squeeze()
    y_tensor_test = torch.from_numpy(y_test.values).type(torch.float).squeeze()
    
    class Malvertising(nn.Module):
        def __init__(self,input,hidden_units):
            super().__init__()
            self.layer_1=nn.Linear(in_features=input,out_features=hidden_units)
            self.layer_2=nn.Linear(in_features=hidden_units,out_features=(hidden_units//2))
            self.layer_3=nn.Linear(in_features=hidden_units//2,out_features=hidden_units//4)
            self.layer_4=nn.Linear(in_features=hidden_units//4,out_features=1)
            self.relu=nn.ReLU()
            self.dropout = nn.Dropout(p=0.3)  
            self.sigmoid=nn.Sigmoid()

        def forward(self,x:torch.Tensor)->torch.Tensor:
            x=self.dropout(self.relu(self.layer_1(x)))
            x=self.dropout(self.relu(self.layer_2(x)))
            x = self.relu(self.layer_3(x))
            x=self.sigmoid(self.layer_4(x))
            return x
            #return self.layer_4(self.relu(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))))
    input_size=x_tensor_train.shape[1]
    model=Malvertising(input=input_size,hidden_units=HIDDEN_UNITS).to(device)



    x_train_tensor=x_tensor_train.to(device)
    y_train_tensor=y_tensor_train.to(device)
    x_test_tensor=x_tensor_test.to(device)
    y_test_tensor=y_tensor_test.to(device)
    return model, x_train_tensor, y_train_tensor, x_test_tensor ,y_test_tensor
        


    
