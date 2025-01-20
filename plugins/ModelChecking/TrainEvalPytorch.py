def accur(y_test,y_pred):
   import torch
   correct=torch.eq(y_test,y_pred).sum().item()
   acc= (correct/len(y_pred))*100
   return acc
def traineval(model,x_train_tensor,y_train_tensor,x_test_tensor,y_test_tensor,LR,EPOCHS):
   import torch.nn as nn
   import torch.optim as optim
   import torch
   import time
   import numpy as np
   from sklearn.utils.class_weight import compute_class_weight
   

  #######################################   Loss & Optimizer #################################
   loss_fn = nn.BCEWithLogitsLoss()
   optimizer=torch.optim.SGD(params=model.parameters(),lr=LR)
   best_acc=0

  #######################################   Looping    #############################################
   torch.manual_seed(42)
   
   start=time.time()
    ################################    Training loop   ####################################
   for epoch in range(EPOCHS):
    model.train()
    y_logits=model(x_train_tensor).squeeze()
    y_pred=torch.round(torch.sigmoid(y_logits))

    loss=loss_fn(y_logits.squeeze(),y_train_tensor.float())
    accuracy=accur(y_train_tensor,y_pred)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    ##########################################  Test    ##########################################""""
    model.eval()
    with torch.inference_mode():
        test_logits=model(x_test_tensor).squeeze()
        test_pred=torch.round(torch.sigmoid(test_logits))

        test_loss=loss_fn(test_logits,y_test_tensor)
        test_accuracy=accur(y_test_tensor,test_pred)
        if test_accuracy>best_acc:
           best_acc=test_accuracy

    if epoch%100==0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {accuracy:.2f}% | Test Loss: {test_loss:.5f} | Test Accuracy: {test_accuracy:.2f}%")
    end=time.time()
   return loss_fn, accur, best_acc

