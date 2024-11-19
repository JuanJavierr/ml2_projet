# %%
import torch
import torch.nn as nn
import numpy as np
import torch.utils
from arima import load_data, get_consumption_for
from typing import Callable
import plotly.express as px
from sklearn.preprocessing import StandardScaler

torch.manual_seed(1)

df = load_data()
df = get_consumption_for(df, "Abitibi", "RÉSIDENTIEL")

df = df[["total_mwh", "month"]].astype("int")

train_df = df["2016":"2022"]
test_df = df["2023":]


scaler = StandardScaler(with_mean=False)
df = scaler.fit_transform(df)

# Convert to tensor
def convert_to_tensor(df):
    scaler = StandardScaler(with_mean=False)
    df = scaler.fit_transform(df)
    df = torch.tensor(df, dtype=torch.float32)

    df[:, 1] = df[:, 1] - 1
    one_hot_months = nn.functional.one_hot(df[:, 1].long(), num_classes=12)
    df = torch.cat((df[:, 0].unsqueeze(1), one_hot_months), dim=1)
    return df.unsqueeze(1), scaler

train_df, _ = convert_to_tensor(train_df)
test_df, scaler = convert_to_tensor(test_df)


#%%

def cosine_lr_schedule(
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float,
    max_lr: float,
) -> Callable[[int], float]:
    def get_lr(t: int) -> float:
        """Outputs the learning rate at step t under the cosine schedule.

        Args:
            t: the current step number

        Returns:
            lr: learning rate at step t

        """

        # assert max_lr >= min_lr >= 0.0
        # assert num_training_steps >= num_warmup_steps >= 0

        # if t <= num_warmup_steps:
        #     # Linear warmup starting from 0 to max_lr
        #     lr = (max_lr / num_warmup_steps) * t
        # elif t >= num_training_steps:
        #     # After training steps, return min_lr
        #     lr = min_lr
        # else:
        #     # Cosine decay from max_lr to min_lr
        #     steps_since_warmup = t - num_warmup_steps
        #     training_steps = num_training_steps - num_warmup_steps
        #     lr = min_lr + 0.5*(max_lr - min_lr) * (1 + np.cos(np.pi * (steps_since_warmup / training_steps)))
        lr = max_lr - (max_lr - min_lr) / num_training_steps * t
        return lr

    return get_lr

class MyLSTM(nn.Module):
    def __init__(self, hidden_size):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(13, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.relu(x)
        x = self.linear(x)
        return x

lstm = MyLSTM(5)
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(lstm.parameters() , lr=0.01)

# %%
EPOCHS = 10_000
lr_schedule = cosine_lr_schedule(100, EPOCHS, 5, 20)
errors = [] # Keep track of errors
best_val_loss = np.inf
steps_since_best = 0
for epoch in range(EPOCHS):
    # lr = lr_schedule(epoch)
    # for g in optimizer.param_groups:
    #     g['lr'] = lr

    optimizer.zero_grad()
    y_pred = lstm(train_df)

    y_pred = y_pred[:-1] # Remove last prediction for which we don't have a target
    
    loss = criterion(y_pred, train_df[1:, 0])

    loss.backward()
    optimizer.step()
    # print(f"Epoch {epoch} Loss: {loss.item()}")

    # Validate
    with torch.no_grad():
        y_pred = lstm(test_df)
        loss = criterion(y_pred, test_df[1:, 0])
        print(f"Validation Loss: {loss.item()}")
        errors.append(loss.item())

        if loss.item() < best_val_loss:
            best_val_loss = loss.item()
            steps_since_best = 0
        else:
            steps_since_best += 1
            if steps_since_best > 100:
                print(f"Early stopping at epoch {epoch}")
                break

# %%
px.line(errors).show()




# %%
scaler.inverse_transform(y_pred)
# %%
