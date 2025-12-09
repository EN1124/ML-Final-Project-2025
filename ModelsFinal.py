
from pytorch_lightning import LightningModule # torch lightning
from typing import Tuple
import torch
import torchmetrics
from torch import Tensor
from torch import nn    # neural network layers
from torch import optim # optimizers



#Base model for neural networks
class BaseModel(LightningModule):
    def __init__(self, input_dim, learning_rate= 0.001):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.accuracy = torchmetrics.classification.BinaryAccuracy() 
        self.model = self.build_model()
       
        
    def build_model(self):
        raise Exception("Not yet implemented")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        return self.model(x)
    
    def loss(self, logits, target):
        loss = nn.BCEWithLogitsLoss()
        return loss(logits.squeeze(-1), target.float())
    
    def shared_step(self, mode:str, batch:Tuple[Tensor, Tensor], batch_index:int):
        x, target = batch
        output = self.forward(x)
        loss = self.loss(output.squeeze(-1), target.float())
        prediction = output.squeeze(-1)
        self.accuracy(prediction, target.float())

        self.log(f"{mode}_step_acc", self.accuracy, prog_bar=True)
        self.log(f"{mode}_step_loss", loss, prog_bar=False)
        return loss
    
    def training_step(self, batch, batch_index):
        return self.shared_step('train', batch, batch_index)
    
    def validation_step(self, batch, batch_index):
        return self.shared_step('val', batch, batch_index)
    
    def test_step(self, batch, batch_index):
        return self.shared_step('test', batch, batch_index)

#Linear classifier (logistic regression, using neural network)
#1 layer neural network

class LinearModel(BaseModel):
    def __init__(self, input_dim):
       super().__init__(input_dim=input_dim)

    def build_model(self):
        #input dimensions= num of features inputted
        #output dimensions= num of possible classifications
        #input tensor x -> linear layer logit/log-odds calculation -> unnormalized probability scores(logits)
        #logits-> sigmoid function -> classification

        in_dim = self.input_dim
        out_dim = 1
        
        return nn.Sequential(
                 nn.Linear(in_dim,out_dim),   
                    )
            
#MLP classifier (final model to be used)
class MLP(BaseModel):
    def __init__(self, input_dim):
       super().__init__(input_dim=input_dim)


    def build_model(self):
        #define dimensions
        in_dim = self.input_dim
        hidden_dim = 100
        out_dim = 1
        feature_2d = 2
        
        #add sequentiallayers
        #input linear layer -> ReLU activiation -> ouput linear layer
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_2d),
            nn.Linear(feature_2d, out_dim),
        )

#sample input function and reset function:
#def Input_UserData(age,income,LoanAmount,education,employment,marital_status,loan_purpose,credit_score,months_employed,num_credit_lines,Interest_Rate,DTI,LoanTerm):
   
   #temp dictionary
   #user_data = {
       #'age': [age],
       #'income': [income],
       #'LoanAmount': [LoanAmount],
        #'Education': [education],
        #'EmploymentType': [employment],
        #'MaritalStatus': [marital_status],
        #'LoanPurpose': [loan_purpose],
        #'CreditScore': [credit_score],
        #'MonthsEmployed': [months_employed],
        #'NumCreditLines': [num_credit_lines],
        #'Interest_Rate': [Interest_Rate],
        #'DTI': [DTI],
        #'LoanTerm': [LoanTerm]
    #}
    # Create single-row DataFrame
    #user_df = pd.DataFrame(user_data)
    
    #return user_df


#def Clear_UserData(user_df):
    #user_df.drop(user_df.index, inplace=True)

    #return user_df

