import torch
import torch.utils.data as torch_data
from torch.distributions import Normal

class NormalYDataset(torch_data.Dataset):
    def __init__(self, df, window_length=126, horizon=5, interval=1, quantile=None):
        self.window_length = window_length
        self.interval = interval
        self.horizon = horizon
        self.quantile=quantile
        
        self.data = df
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()
        self.standard_normal=Normal(loc=0., scale=1.)

    def __getitem__(self, index):
        hi = self.x_end_idx[index]
        lo = hi - self.window_length
        x_data = self.data[lo: hi]
        y_data = self.data[hi+1:hi + self.horizon+1]
        
        ### exclude stocks listed for trading for less than 50% days of the observation window
        ### exclude stocks having been delisted by the last day of the observation window
        x_data = x_data.loc[:,(x_data.count()>=len(x_data)/2)&(~x_data.iloc[-1].isna())].fillna(0)
        y_data = y_data.loc[:,x_data.columns].mean(0)
        
        x = torch.tensor(x_data.values,dtype=torch.float)
        y = torch.tensor(y_data.values,dtype=torch.float)
        yrank=torch.zeros_like(y,dtype=torch.float)
        yrank[torch.argsort(y)]=self.standard_normal.icdf(torch.arange(1,len(y)+1,dtype=torch.float)/(len(y)+1))
        
        if self.quantile:
            x=x[:,yrank>yrank.quantile(self.quantile)]
            yrank=yrank[yrank>yrank.quantile(self.quantile)]
            
        return x, yrank

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        x_index_set = range(self.window_length, self.df_length - self.horizon, self.interval)
        x_end_idx = [x_index_set[j] for j in range(len(x_index_set))]
        return x_end_idx
    

class TestDataset(torch_data.Dataset):
    def __init__(self, df, window_length=126, horizon=5, interval=1,data_type='return'):
        self.window_length = window_length
        self.interval = interval
        self.horizon = horizon
        self.data_type=data_type
        
        self.data = df
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()

    def __getitem__(self, index):
        hi = self.x_end_idx[index]
        lo = hi - self.window_length
        x_data = self.data[lo: hi]
        
        ### exclude stocks listed for trading for less than 50% days of the observation window
        ### exclude stocks having been delisted by the last day of the observation window
        x_data = x_data.loc[:,(x_data.count()>=len(x_data)/2)&(~x_data.iloc[-1].isna())]
        if self.data_type=='return':
            x_data=x_data.fillna(0)
        elif self.data_type=='price':
            x_data=x_data.bfill()
        x = torch.tensor(x_data.values,dtype=torch.float)
        
        return x, x_data.columns,self.data.index[hi + self.horizon]

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        x_index_set = range(self.window_length, self.df_length - self.horizon, self.interval)
        x_end_idx = [x_index_set[j] for j in range(len(x_index_set))]
        return x_end_idx