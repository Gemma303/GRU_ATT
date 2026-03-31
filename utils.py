import numpy as np
import pandas as pd
import time
import statsmodels.api as sm
from linearmodels import PanelOLS
from soft_dtw_cuda import SoftDTW

import torch
import torch.nn as nn
import torch.utils.data as torch_data
from DataLoader import NormalYDataset,TestDataset
from model import gru_gnn,gru

class CorrLoss(nn.Module):
    def __init__(self):
        super(CorrLoss, self).__init__()

    def forward(self, output, target):
        output_dm=output-torch.mean(output)
        target_dm=target-torch.mean(target)
        return -torch.sum(output_dm*target_dm)/torch.sqrt(torch.sum(output_dm**2)*torch.sum(target_dm**2))

def train(train_valid_data,model_state_path,args):
    
    random_seed=args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    long_data = NormalYDataset(train_valid_data, window_length=args.window_length, horizon=args.horizon, interval=args.interval, quantile=1-args.long_pct)
    all_data = NormalYDataset(train_valid_data, window_length=args.window_length, horizon=args.horizon, interval=args.interval)
    
    indices = torch.randperm(len(all_data)).tolist()
    split = int(0.7 * len(indices))

    train_indices = indices[:split]
    valid_indices = indices[split:]

    train_set_long = torch_data.Subset(long_data, train_indices)
    train_set = torch_data.Subset(all_data, train_indices)
    valid_set = torch_data.Subset(all_data, valid_indices)

    train_loader_long = torch_data.DataLoader(train_set_long, shuffle=True)
    train_loader = torch_data.DataLoader(train_set, shuffle=True)
    valid_loader = torch_data.DataLoader(valid_set)
    
    if args.model=='gru_gnn':
        model=gru_gnn(gru_lengths=args.gru_lengths,k_dim=args.k_dim,dropout_rate=args.dropout_rate,device=args.device).to(args.device)
    elif args.model=='gru':
        model=gru(gru_lengths=args.gru_lengths,device=args.device).to(args.device)
    else:
        raise ValueError("model needs to be gru_gnn or gru")
    forecast_loss = CorrLoss()
    my_optim = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    my_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=my_optim, T_max=args.epoch,eta_min=1e-4) 
    
    model.train()
    for epoch in range(args.epoch_long):
        epoch_start_time = time.time()
        loss_total = 0
        for inputs, target in train_loader_long:    
            inputs=inputs.squeeze().to(args.device)
            target=target.squeeze().to(args.device)
        
            my_optim.zero_grad()
            forecast=model(inputs)
            loss = forecast_loss(forecast, target)
            loss.backward()       
            my_optim.step()
            loss_total += float(loss)
                
        average_loss_train=loss_total/len(train_set)
        print(f'LongEndTraining Epoch {epoch}, time{time.time()-epoch_start_time}, trainset loss{average_loss_train}')
        
    last_valid_loss = np.inf
    best_valid_loss = np.inf
    valid_loss_non_decrease_count = 0  
    train_loss_epochs=[]
    valid_loss_epochs=[]
        
    for epoch in range(args.epoch):
        model.train()
        epoch_start_time = time.time()
        total_loss_train = 0
        total_loss_valid = 0
        for inputs, target in train_loader:    
            inputs=inputs.squeeze().to(args.device)
            target=target.squeeze().to(args.device)
        
            my_optim.zero_grad()
            forecast=model(inputs)
            loss = forecast_loss(forecast, target)
            loss.backward()       
            my_optim.step()
            total_loss_train += float(loss)
                
        average_loss_train=total_loss_train/len(train_set)
        train_loss_epochs.append(average_loss_train)
        
        model.eval()
        with torch.no_grad():
            for inputs, target in valid_loader:    
                inputs=inputs.squeeze().to(args.device)
                target=target.squeeze().to(args.device)
                forecast=model(inputs)
                loss = forecast_loss(forecast, target)
                total_loss_valid += float(loss)
        average_loss_valid=total_loss_valid/len(valid_set)
        valid_loss_epochs.append(average_loss_valid)
        print(f'AllTraining Epoch {epoch}, time{time.time()-epoch_start_time}, trainset loss{average_loss_train}, validset loss{average_loss_valid}')
            
        if (epoch+1) % args.exponential_decay_epoch == 0:
            my_lr_scheduler.step()
            
        if average_loss_valid < last_valid_loss:
            valid_loss_non_decrease_count = 0
            #torch.save(model.state_dict(), model_state_path)
        else:
            valid_loss_non_decrease_count += 1 
        last_valid_loss=average_loss_valid

        if average_loss_valid<best_valid_loss:
            torch.save(model.state_dict(), model_state_path)
            best_valid_loss=average_loss_valid
        
        if valid_loss_non_decrease_count >= args.early_stop_epoch:
            return train_loss_epochs,valid_loss_epochs
        # if epoch == args.epoch-1 :
        #     torch.save(model.state_dict(), model_state_path)
    return train_loss_epochs,valid_loss_epochs
        
def test(test_data,return_data,model_state_path,test_result_path,test_portfolio_path,args):
    weekly_return=(np.exp(np.log(1+return_data['DailyReturn']).unstack().rolling(args.horizon,min_periods=1).sum())-1).stack().dropna().sort_index().to_frame().rename(columns={0:'WeeklyReturn'})
    test_set = TestDataset(test_data, window_length=args.window_length, horizon=args.horizon, interval=args.interval)
    if args.model=='gru_gnn':
        model=gru_gnn(gru_lengths=args.gru_lengths,k_dim=args.k_dim,dropout_rate=args.dropout_rate,device=args.device).to(args.device)
    elif args.model=='gru':
        model=gru(gru_lengths=args.gru_lengths,device=args.device).to(args.device)
    else:
        raise ValueError("model needs to be gru_gnn or gru")
    model.load_state_dict(torch.load(model_state_path,weights_only=True))
    model.eval()
    
    test_list=[]
    with torch.no_grad():
        for i in range(len(test_set)):   
            curr_test_data=test_set[i]
            inputs,secucodes,date=curr_test_data[0].squeeze().to(args.device),curr_test_data[1],curr_test_data[2]

            forecast = model(inputs)
            forecast = pd.DataFrame(forecast.cpu().numpy(),columns=['Signal']).assign(SecuCode=secucodes).assign(DataDate=date)
            test_list.append(forecast)
    signal_df=pd.concat(test_list).set_index(['DataDate','SecuCode']).sort_index()
    
    compare=signal_df.merge(weekly_return,left_index=True,right_index=True)
    RankCorr=compare.groupby('DataDate').apply(lambda df:df.corr(method='spearman').iloc[0,1])
    RankIC=RankCorr.mean()
    RankICIR=RankIC/RankCorr.std()
    compare['Group']=compare.groupby('DataDate')['Signal'].apply(lambda x:pd.qcut(x,10,[f'Group{i}' for i in range(1,11)]))
    group_weekly_return=compare.groupby(['DataDate','Group'])['WeeklyReturn'].mean().unstack()
    signal_df.to_pickle(test_result_path)
    group_weekly_return.to_pickle(test_portfolio_path)
    
    return RankIC,RankICIR,group_weekly_return

def construct_expo_weights(half_life, length):
    if half_life == 0:
        exponential_weights = np.ones(length) / length
    else:
        exponential_lambda = 0.5 ** (1 / half_life)
        exponential_weights = (
            1 - exponential_lambda
        ) * exponential_lambda ** np.arange(length - 1, -1, -1)
        exponential_weights = exponential_weights / np.sum(exponential_weights)
    return exponential_weights

def get_weighted_return_avg(return_data,start,end,min_periods,horizon,signal_name):
    period_length=end-start
    exp_w=construct_expo_weights(period_length//2,period_length)
    return_data=return_data['DailyReturn'].unstack().shift(start)
    signal=pd.DataFrame(index=return_data.index[end-1:],columns=return_data.columns)
    
    for i in range(end,len(return_data)):
        signal.iloc[i-end]=return_data.iloc[i-period_length:i].multiply(exp_w,axis=0).divide((~return_data.iloc[i-period_length:i].isna()).multiply(exp_w,axis=0).sum()).sum(min_count=min_periods)
    signal=signal.shift(horizon+1).stack().dropna().to_frame().rename(columns={0:signal_name})
    return signal
        
def get_return_vol(return_data,start,end,min_periods,horizon,signal_name):
    period_length=end-start
    signal=return_data['DailyReturn'].unstack().shift(start).rolling(period_length,min_periods=min_periods).std().iloc[end-1:].shift(horizon+1).stack().dropna().to_frame().rename(columns={0:signal_name})
    return signal


def panelols_regression(data,Y,Xs,reg_type='univariate'):
    if reg_type=='univariate':
        coefs=[]
        pvals=[]
        rsqs=[]
        for X in Xs:
            exog = sm.tools.tools.add_constant(data[X])
            endog = data[Y]
            model_ols = PanelOLS(endog, exog, entity_effects = False, time_effects = False) 
            results = model_ols.fit(cov_type='kernel', kernel='bartlett') 
            coefs.append(results.params[-1])
            pvals.append(results.pvalues[-1])
            rsqs.append(results.rsquared)
        output=pd.DataFrame(np.array([coefs,pvals]).T.round(4),index=Xs,columns=['Coefficient','P-value']).assign(RSquared=rsqs)
        return output
    elif reg_type=='multivariate':
        exog = sm.tools.tools.add_constant(data[Xs])
        endog = data[Y]
        model_ols = PanelOLS(endog, exog, entity_effects = False, time_effects = False) 
        results = model_ols.fit(cov_type='kernel', kernel='bartlett') 
        output=pd.DataFrame(np.array([results.params,results.pvalues]).T.round(4),index=['const']+Xs,columns=['Coefficient','P-value']).assign(RSquared=results.rsquared)
        return output
    else:
        raise ValueError("reg_type needs to be univariate or multivariate")
    
def get_reg_results(signal_df,return_data,horizon):
    weekly_return=(np.exp(np.log(1+return_data['DailyReturn']).unstack().rolling(horizon,min_periods=1).sum())-1).stack().dropna().sort_index().to_frame().rename(columns={0:'WeeklyReturn'})
    trad_inds=['WAVG','MAVG','LAVG','VOL']
    data=(weekly_return
          .merge(signal_df,left_index=True,right_index=True)
          .merge(get_weighted_return_avg(return_data,0,5,5,horizon,'WAVG'),left_index=True,right_index=True)
          .merge(get_weighted_return_avg(return_data,0,21,11,horizon,'MAVG'),left_index=True,right_index=True)
          .merge(get_weighted_return_avg(return_data,21,126,53,horizon,'LAVG'),left_index=True,right_index=True)
          .merge(get_return_vol(return_data,0,126,64,horizon,'VOL'),left_index=True,right_index=True)
          .rename(columns={'Signal':'gru_gnn'})
          )
    data=data.groupby('DataDate').apply(lambda df:df.subtract(df.mean(),1).divide(df.std(),1)).swaplevel().sort_index()
    
    res1=panelols_regression(data,'WeeklyReturn',['gru_gnn']+trad_inds,reg_type='univariate')
    res2=panelols_regression(data,'gru_gnn',trad_inds,reg_type='multivariate')
    res3=panelols_regression(data,'WeeklyReturn',['gru_gnn']+trad_inds,reg_type='multivariate')
    return res1,res2,res3
    
    
    
def get_patterns():
    linear_dict={'double_bottom':[(0,0),(32,-0.1),(63,-0.05),(94,-0.1),(126,0)],
                'double_top':[(0,0),(32,0.1),(63,0.05),(94,0.1),(126,0)],
                'head_shoulders_bottom':[(0,0),(21,-0.1),(32,-0.05),(63,-0.2),(94,-0.05),(105,-0.1),(126,0)],
                'head_shoulders_top':[(0,0),(21,0.1),(32,0.05),(63,0.2),(94,0.05),(105,0.1),(126,0)],
                'ascend_triangle':[(0,0),(24,0.1),(44,0.02),(64,0.1),(79,0.04),(94,0.1),(104,0.06),(126,0.15)],
                'descend_triangle':[(0,0),(24,-0.1),(44,-0.02),(64,-0.1),(79,-0.04),(94,-0.1),(104,-0.06),(126,-0.15)],
                'bull_rectangle':[(0,0),(18,0.1),(35,0.02),(52,0.1),(69,0.02),(86,0.1),(103,0.02),(126,0.14)],
                'bear_rectangle':[(0,0),(18,-0.1),(35,-0.02),(52,-0.1),(69,-0.02),(86,-0.1),(103,-0.02),(126,-0.14)],
                'bull_pennant':[(0,0),(20,0.1),(39,0.01),(56,0.085),(76,0.025),(93,0.07),(107,0.04),(126,0.08)],
                'bear_pennant':[(0,0),(20,-0.1),(39,-0.01),(56,-0.085),(76,-0.025),(93,-0.07),(107,-0.04),(126,-0.08)],
                'falling_wedge':[(0,0),(18,0.05),(36,-0.01),(54,0.025),(72,-0.02),(90,0),(108,-0.03),(126,-0.005)],
                'rising_wedge':[(0,0),(18,-0.05),(36,0.01),(54,-0.025),(72,0.02),(90,0),(108,0.03),(126,0.005)]}
 
    nonlinear_dict={'rounding_bottom':-np.sqrt(0.55**2-(np.linspace(0,1,127)-0.45)**2)/5,
                    'rounding_top':np.sqrt(0.55**2-(np.linspace(0,1,127)-0.45)**2)/5}
    
    
    pattern_df=pd.DataFrame(index=range(127),columns=list(linear_dict.keys())+list(nonlinear_dict.keys()))
    for key,value in linear_dict.items():
        xs,ys=zip(*value)
        pattern_df[key]=np.interp(range(0,127),list(xs),list(ys))
        
    for key,value in nonlinear_dict.items():
        pattern_df[key]=value
        
    return pattern_df
    
def get_pattern_stocks(test_data,args,pattern_path):
    k=args.k_pattern
    pattern_df=get_patterns()
    if args.device=='cuda':
        sdtw =SoftDTW(use_cuda=True, gamma=0.1)
    else:
        sdtw =SoftDTW(use_cuda=False, gamma=0.1)
    test_set = TestDataset(test_data, window_length=args.window_length+1, horizon=args.horizon, interval=args.interval,data_type='price')
    
    pattern_stocks_list=[]
    for col in pattern_df.columns:
        pattern=pattern_df[col].values
        pattern=torch.tensor((pattern-pattern.mean())/pattern.std())
        
        for i in range(len(test_set)):
            curr_test_data=test_set[i]
            x,secucodes,date=curr_test_data[0].squeeze().to(args.device),curr_test_data[1],curr_test_data[2]
            
            x=((x-x.mean(0))/x.std(0)).T.unsqueeze(-1)
            pattern_reshape=torch.tile(pattern,(x.shape[0],1)).unsqueeze(-1).to(args.device)

            distances=sdtw(pattern_reshape,x)
            _, smallest_indices = torch.topk(distances, k=k, largest=False)
            smallest_indices=smallest_indices.cpu().tolist()
            
            curr_pattern_stocks=(pd.DataFrame(secucodes[smallest_indices],columns=['SecuCode'])
                        .assign(DataDate=date)
                        .assign(Pattern=col)
                        .assign(Rank=range(k)))
            pattern_stocks_list.append(curr_pattern_stocks)
            
    pattern_stocks=pd.concat(pattern_stocks_list)
    pattern_stocks.to_pickle(pattern_path)
    return pattern_stocks

def test_pattern_stocks(test_data,return_data,model_state_path,args,pattern_stocks):
    k=args.k_pattern_test
    pattern_stocks=(pattern_stocks
                    .sort_values(['DataDate','Rank','Pattern'])
                    .drop_duplicates(['DataDate','SecuCode'],keep='first')
                    .assign(Rank_New=lambda df:df.groupby(['DataDate','Pattern'])['SecuCode'].transform(lambda x:list(range(len(x)))))
                    .query(f"Rank_New<{k}")
                    .set_index(['DataDate','Pattern'])[['SecuCode']]
                    .sort_index()
                    )
    pattern_stock_counts=pattern_stocks.groupby(['DataDate','Pattern'])['SecuCode'].count().loc[lambda x:x!=k]
    if len(pattern_stock_counts)>0:
        pattern_stocks=pattern_stocks[~pattern_stocks.index.get_level_values(0).isin(pattern_stock_counts.index.get_level_values(0).drop_duplicates().tolist())]
    pattern_dates=pattern_stocks.index.get_level_values(0).drop_duplicates().tolist()
    
    weekly_return=(np.exp(np.log(1+return_data['DailyReturn']).unstack().rolling(args.horizon,min_periods=1).sum())-1).stack().dropna().sort_index().to_frame().rename(columns={0:'WeeklyReturn'})
    test_set = TestDataset(test_data, window_length=args.window_length, horizon=args.horizon, interval=args.interval)
    model=gru_gnn(gru_lengths=args.gru_lengths,k_dim=args.k_dim,dropout_rate=args.dropout_rate,device=args.device).to(args.device)
    model.load_state_dict(torch.load(model_state_path,weights_only=True))
    model.eval()
    
    test_list=[]
    with torch.no_grad():
        for i in range(len(test_set)):   
            curr_test_data=test_set[i]
            inputs,secucodes,date=curr_test_data[0].squeeze().to(args.device),curr_test_data[1],curr_test_data[2]

            if date in pattern_dates:
            
                pattern_dummy=pd.Series(secucodes).isin(pattern_stocks.loc[date]['SecuCode'].tolist())
                inputs=inputs[:,pattern_dummy]
                secucodes=secucodes[pattern_dummy]

                forecast = model(inputs)
                forecast = pd.DataFrame(forecast.cpu().numpy(),columns=['Signal']).assign(SecuCode=secucodes).assign(DataDate=date)
                test_list.append(forecast)
    signal_df=pd.concat(test_list).set_index(['DataDate','SecuCode']).sort_index()
    
    compare=(signal_df
             .merge(weekly_return,left_index=True,right_index=True)
             .merge(pattern_stocks.reset_index(level=-1).set_index('SecuCode',append=True),left_index=True,right_index=True)
             .assign(return_label=lambda df:df.groupby('DataDate')['WeeklyReturn'].transform(lambda x:x>x.median()))
             .assign(signal_label=lambda df:df.groupby('DataDate')['Signal'].transform(lambda x:x>x.median()))
             .assign(pattern_label=lambda df:df.Pattern.str.contains(r'bottom|ascend|bull|falling')))
    
    cm_signal_pattern = ((pd.crosstab(compare['signal_label'],compare['pattern_label'],rownames=['gru_gnn'],colnames=['chart_pattern'])/len(compare))
                        .rename(columns={False:'Negative',True:'Positive'},index={False:'Negative',True:'Positive'}))
    acc_pattern=compare.groupby(['DataDate','Pattern']).apply(lambda df:(df['return_label']==df['pattern_label']).sum()/len(df))
    acc_pattern_all=compare.groupby(['DataDate']).apply(lambda df:(df['return_label']==df['pattern_label']).sum()/len(df))
    acc_signal=compare.groupby(['DataDate','Pattern']).apply(lambda df:(df['return_label']==df['signal_label']).sum()/len(df))
    acc_signal_all=compare.groupby(['DataDate']).apply(lambda df:(df['return_label']==df['signal_label']).sum()/len(df))
    
    acc_pattern_mean=pd.concat([acc_pattern.groupby('Pattern').mean(),pd.Series([acc_pattern_all.mean()],index=pd.Index(['Total'],name='Pattern'))])
    acc_pattern_se=pd.concat([acc_pattern.groupby('Pattern').sem(),pd.Series([acc_pattern_all.sem()],index=pd.Index(['Total'],name='Pattern'))])
    
    acc_signal_mean=pd.concat([acc_signal.groupby('Pattern').mean(),pd.Series([acc_signal_all.mean()],index=pd.Index(['Total'],name='Pattern'))])
    acc_signal_se=pd.concat([acc_signal.groupby('Pattern').sem(),pd.Series([acc_signal_all.sem()],index=pd.Index(['Total'],name='Pattern'))])
    
    acc_compare=pd.concat([acc_pattern_mean,acc_pattern_se,acc_signal_mean,acc_signal_se],axis=1)
    acc_compare.columns=pd.MultiIndex.from_product([['chart_pattern','gru_gnn'],['mean','se']])

    return cm_signal_pattern,acc_compare

def dd_analysis(portfolio,k=3):
    portfolio['NAV']=(1+portfolio['Return']).cumprod()
    portfolio['DD']=(portfolio['NAV']-portfolio['NAV'].cummax())/portfolio['NAV'].cummax()
    portfolio['DD_Flag']=0
    
    begins=[]
    ends=[]
    depths=[]
    for _ in range(k):
        if (portfolio['DD']<0).sum()==0:
            break
        midx=portfolio.loc[~portfolio['DD_Flag'].astype(bool)]['DD'].idxmin()
        begin=portfolio.loc[:midx].query("DD>=0").index[-1]+1
        end_df=portfolio.loc[midx:].query("DD>=0")
        if len(end_df)>0:
            end=end_df.index[0]-1
        else:
            end=len(portfolio)-1
        portfolio.loc[begin:end,'DD_Flag']=1
        begins.append(portfolio.loc[begin]['DataDate'])
        ends.append(portfolio.loc[end]['DataDate'])
        depths.append(portfolio.loc[midx]['DD'])
    output=pd.DataFrame([begins,ends,depths],index=['StartDate','EndDate','DrawDown']).T
    return portfolio,output