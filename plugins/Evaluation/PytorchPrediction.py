def prediction_pytorch(model,x_test_tensor,y_test_tensor,loss_fn,accur):
    import torch
    torch.manual_seed(42)
    model.eval()
    with torch.inference_mode():
        y_pred_final = torch.round(torch.sigmoid(model(x_test_tensor))).squeeze()
        test_loss_final = loss_fn(y_pred_final, y_test_tensor)
        y_pred_labels = (y_pred_final > 0.5).float()
        accuracy = accur(y_test_tensor, y_pred_labels)
        print(f"Final Test Loss: {test_loss_final.item():.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
   