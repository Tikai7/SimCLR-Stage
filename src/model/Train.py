import torch
import os
from tqdm import tqdm
from model.BERT import BertEncoder
from transformers import BertTokenizer
class Trainer:
    """
    Trainer class to train the SimCLR model specifically
    """

    def __init__(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bert = BertEncoder()
        self.bert.to(self.device)
        self.bert.eval()
        self.history = {
            "params" : {
                "lr": None,
                "epochs" : None,
                "is_val" : None,
                "model" : None,
            },
            "validation" : {
                "loss" : []
            },
            "train" : {
                "loss" :  []
            }
        }

    def set_loss(self, loss_fn):
        """
        Set the loss function for the model
        The loss function should be a callable function
        """
        self.loss_fn = loss_fn
        return self

    def set_model(self, model, name="SimCLR"):
        """
            Set the model for the trainer   
        """
        self.model = model
        self.model.to(self.device)
        self.history['params']['model'] = name
        print(f"[INFO] Model's device is : {self.device}")
        return self

    def set_optimizer(self, optimizer : torch.optim.Optimizer):
        """
        Set the optimizer for the model
        The optimizer should be a callable function
        """
        self.optimizer = optimizer
        return self

    def save(self, model_path : str = "model.pth", history_path : str = "history.txt"):
        """
        Save the model and the history of the training
        """
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.history, history_path)

    def get_history(self, history_path :str):
        """
        Get the history of the training
        """
        assert os.path.exists(history_path), "[ERROR] history path does not exist"
        return torch.load(history_path)
    
    def get_model(self, model_path):
        """
        Get the model state dict
        """
        assert os.path.exists(model_path), "[ERROR] path does not exist"
        return torch.load(model_path)
    
    def fit(self, train_data, validation_data=None, learning_rate=1e-4, epochs=1, verbose=True, weight_decay=1e-6):
        """
        Fit the model to the data
        First call : 
        .set_model() : to set the model
        .set_optimizer() : to set the optimizer
        .set_loss() : to set the loss function

        You can find the history of the training in self.history
        """

        assert self.model is not None, "[ERROR] set or load the model first throught .set_model() or .load_model()"
        assert self.optimizer is not None, "[ERROR] set the optimizer first throught .set_optimizer()"
        assert self.loss_fn is not None, "[ERROR] set the loss function first throught .set_loss()"

        self.optimizer = self.optimizer(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.history["params"]["lr"] = learning_rate
        self.history["params"]["epochs"] = epochs
        self.history["params"]["is_val"] = True if validation_data is not None else False


        best_loss = float('inf')
        for epoch in tqdm(range(epochs)):
            train_loss, val_loss = None, None

            train_loss = self._train(train_data)
            val_loss = self._validate(validation_data)

            self._print_epoch(epoch, train_loss, val_loss, verbose)

            self.history['train']['loss'].append(train_loss)
            self.history['validation']['loss'].append(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = self.model 

        self.model = best_model
        return self.model

    def _train(self, train_data):
        losses = 0
        self.model.train()
        for data in train_data:
            output = None

            batch_x, batch_y, context_x, context_y = data
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            context_x = self.tokenizer(list(context_x), padding=True, return_tensors='pt', add_special_tokens=True)
            context_y = self.tokenizer(list(context_y), padding=True, return_tensors='pt', add_special_tokens=True)
            output = self.model(batch_x, batch_y, context_x, context_y)

            Z1, Z2 = output["projection_head"]
            loss = self.loss_fn(Z1,Z2)
            
            losses += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return losses/len(train_data)

    def _validate(self, validation_data):
        losses = 0
        self.model.eval()
        with torch.no_grad():
            for data in validation_data:
                output = None

                batch_x, batch_y, context_x, context_y = data
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                context_x = self.tokenizer(list(context_x), padding=True, return_tensors='pt', add_special_tokens=True)
                context_y = self.tokenizer(list(context_y), padding=True, return_tensors='pt', add_special_tokens=True)
                output = self.model(batch_x, batch_y, context_x, context_y)

                Z1, Z2 = output["projection_head"]
                loss = self.loss_fn(Z1,Z2)
                losses += loss.item()

        return losses/len(validation_data)
    
    def _print_epoch(self, epoch, train_loss, val_loss, verbose):
        if verbose :
            print(f"Epoch : {epoch}, Train loss : {train_loss}, Validation loss : {val_loss}")

