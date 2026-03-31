import os
import pandas as pd
import numpy as np
from datetime import datetime
from utils import train,test,get_reg_results,get_patterns,get_pattern_stocks,test_pattern_stocks,dd_analysis
import argparse
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=52)
parser.add_argument('--train_length', type=int, default=1260)
parser.add_argument('--valid_length', type=int, default=447)
parser.add_argument('--window_length', type=int, default=126)
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--interval', type=int, default=1)
parser.add_argument('--long_pct', type=int, default=0.25)
parser.add_argument('--gru_lengths', type=list, default=[126,20,60])
parser.add_argument('--k_dim', type=int, default=8)
parser.add_argument('--dropout_rate', type=float, default=0.2)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--w_decay', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--epoch_long', type=int, default=5)
parser.add_argument('--exponential_decay_epoch', type=int, default=1)
parser.add_argument('--early_stop_epoch', type=int, default=10)
parser.add_argument('--k_pattern', type=int, default=50)
parser.add_argument('--k_pattern_test', type=int, default=20)
parser.add_argument('--top_drawdown_periods', type=int, default=3)
parser.add_argument('--price_data', type=str, default='closeprice_adj.pickle')
parser.add_argument('--return_data', type=str, default='dailyreturn.pickle')
parser.add_argument('--model_state', type=str, default='gru_gnn.pth')
parser.add_argument('--test_results', type=str, default='test_forecast.pickle')
parser.add_argument('--test_portfolio', type=str, default='test_portfolio.pickle')
parser.add_argument('--pattern_data', type=str, default='pattern_data.pickle')
parser.add_argument('--dd_periods', type=str, default='dd_periods.pickle')
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--signal_compare', type=bool, default=True)
parser.add_argument('--pattern_compare', type=bool, default=True)
parser.add_argument('--model', type=str, default='gru_gnn')


args = parser.parse_args()
if __name__ == '__main__':
    
    model_state_path=os.path.join('model_states',args.model_state)
    test_result_path=os.path.join('results',args.test_results)
    test_portfolio_path=os.path.join('results',args.test_portfolio)
    pattern_path=os.path.join('data',args.pattern_data)
    dd_path=os.path.join('results',args.dd_periods)
    
    price_data=pd.read_pickle(os.path.join('data',args.price_data))
    test_data_price=price_data['ClosePrice_adj'].unstack().sort_index().iloc[args.train_length+args.valid_length-args.window_length-1:]

    return_data=pd.read_pickle(os.path.join('data',args.return_data))
    return_data_pivot=return_data['DailyReturn'].unstack().sort_index()
    normdata=return_data_pivot.subtract(return_data_pivot.mean(1),0).divide(return_data_pivot.std(1),0)
    
    train_valid_data=normdata.iloc[:args.train_length+args.valid_length]
    test_data=normdata.iloc[args.train_length+args.valid_length-args.window_length:]
    
    if args.train:
        print("Training begins.")
        start_time = datetime.now().timestamp()
        train_loss_epochs,valid_loss_epochs = train(train_valid_data,model_state_path,args)

        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(go.Scatter(x=list(range(1,len(train_loss_epochs)+1)), y=train_loss_epochs,mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(1,len(valid_loss_epochs)+1)), y=valid_loss_epochs,mode='lines'), row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Training Loss", row=1, col=1)
        fig.update_yaxes(title_text="Validation Loss", row=1, col=2)
        fig.update_layout(width=1000, height=500,showlegend=False)
        with open("results/train_results.html", "w") as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        
        end_time = datetime.now().timestamp()
        print(f'Training took {np.ceil((end_time - start_time) / 60)} minutes.')
        
    if args.evaluate:
        print("Evaluation begins.")
        start_time = datetime.now().timestamp()
        RankIC,RankICIR,group_weekly_return=test(test_data,return_data,model_state_path,test_result_path,test_portfolio_path,args)
        LongPortfolio,LSPortfolio=group_weekly_return['Group10'].iloc[::args.horizon],(group_weekly_return['Group10']-group_weekly_return['Group1']).iloc[::args.horizon]

        fig_decile = make_subplots(rows=1, cols=2, subplot_titles=("Decile Portfolio Return", "Decile Portfolio Volatility"))
        fig_decile.add_trace(go.Bar(x=list(range(1,11)), y=group_weekly_return.mean()*252/5), row=1, col=1)
        fig_decile.add_trace(go.Bar(x=list(range(1,11)), y=group_weekly_return.std()*np.sqrt(252/5)), row=1, col=2)
        fig_decile.update_xaxes(title_text="Signal Decile", row=1, col=1)
        fig_decile.update_xaxes(title_text="Signal Decile", row=1, col=2)
        fig_decile.update_yaxes(title_text="Annualized Average Return", row=1, col=1)
        fig_decile.update_yaxes(title_text="Annualized Volatility", row=1, col=2)
        fig_decile.update_layout(width=1000, height=500,showlegend=False)

        LSPortfolio_DD=LSPortfolio.reset_index().rename(columns={0:'Return'})
        LSPortfolio_DD,dd_results=dd_analysis(LSPortfolio_DD,k=args.top_drawdown_periods)
        with open("results/test_results.html", "w") as f:
            f.write(f"RankIC: {np.round(RankIC,4)}, RankICIR: {np.round(RankICIR,4)}<br>")
            f.write("Long Only Portfolio:<br>")
            f.write(f"Annualized Average Return: {np.round(LongPortfolio.mean()*252/5,4)}, Annualized Sharpe Ratio: {np.round(LongPortfolio.mean()/LongPortfolio.std()*np.sqrt(252/5),4)}<br>")
            f.write("Long Short Portfolio:<br>")
            f.write(f"Annualized Average Return: {np.round(LSPortfolio.mean()*252/5,4)}, Annualized Sharpe Ratio: {np.round(LSPortfolio.mean()/LSPortfolio.std()*np.sqrt(252/5),4)}<br>")
            f.write(fig_decile.to_html(full_html=False, include_plotlyjs='cdn'))

            f.write("Long Short Portfolio DrawDown Analysis:<br>")
            if len(dd_results)==0:
                f.write(f"No drawdown periods.<br>")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=LSPortfolio_DD['DataDate'], y=LSPortfolio_DD['NAV'], mode='lines'))
                fig.update_yaxes(range=[LSPortfolio_DD['NAV'].min(), LSPortfolio_DD['NAV'].max()],title_text="Long Short Portfolio NAV")
                fig.update_xaxes(title_text="Date")
                fig.update_layout(width=1000, height=500)
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=LSPortfolio_DD['DataDate'], y=LSPortfolio_DD['NAV'], mode='lines'))
                for i in range(len(dd_results)):
                    begin=dd_results['StartDate'].iloc[i]
                    end=dd_results['EndDate'].iloc[i]
                    fig.add_shape(type="rect",x0=begin,x1=end,y0=0, y1=1, yref="paper", fillcolor="pink",opacity=0.5,line_width=0)
                    fig.add_shape(type="line",x0=end,x1=end,y0=0,y1=1,yref="paper",line=dict(color="pink", width=2, dash="solid"),opacity=0.8)
                    fig.add_shape(type="line",x0=begin,x1=begin,y0=0,y1=1,yref="paper",line=dict(color="pink", width=2, dash="solid"),opacity=0.8)
                fig.update_yaxes(range=[LSPortfolio_DD['NAV'].min(), LSPortfolio_DD['NAV'].max()],title_text="Long-Short Portfolio NAV")
                fig.update_xaxes(title_text="Date")
                fig.update_layout(width=1000, height=500)
                
                f.write(f"Top {len(dd_results)} drawdown periods.<br>")
                f.write(dd_results.to_html())
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        end_time = datetime.now().timestamp()
        print(f'Evaluation took {np.ceil((end_time - start_time) / 60)} minutes.')
        
    if args.signal_compare:
        print("Traditional trend signal evaluation begins.")
        start_time = datetime.now().timestamp()
        signal_df=pd.read_pickle(test_result_path)
        res1,res2,res3=get_reg_results(signal_df,return_data,args.horizon)
        with open("results/signal_compare_results.html", "w") as f:
            f.write(f"Univariate regression of weekly return over each signal <br>")
            f.write(res1.to_html())
            f.write("<br><br>")
            f.write(f"Multivariate regression of gru_gnn signal over traditional trend signals: r-squared {np.round(res2['RSquared'].iloc[0],6)} <br>")
            f.write(res2[['Coefficient','P-value']].to_html())
            f.write("<br><br>")
            f.write(f"Multivariate regression of weekly return over all signals: r-squared {np.round(res3['RSquared'].iloc[0],6)} <br>")
            f.write(res3[['Coefficient','P-value']].to_html())
        end_time = datetime.now().timestamp()
        print(f'Traditional trend signal evaluation took {np.ceil((end_time - start_time) / 60)} minutes.')
        
    if args.pattern_compare:
        print("Chart signal evaluation begins.")
        start_time = datetime.now().timestamp()
        print("Getting pattern stocks.")
        if os.path.exists(pattern_path):
            pattern_stocks=pd.read_pickle(pattern_path)
        else:
            pattern_stocks=get_pattern_stocks(test_data_price,args,pattern_path)
        print("Evaluating pattern stocks.")
        cm_signal_pattern,acc_compare=test_pattern_stocks(test_data,return_data,model_state_path,args,pattern_stocks)
        patterns=get_patterns()
        acc_compare_show=(acc_compare.loc[list(patterns.columns)+['Total']].style.apply(lambda row:['color: red' if row.iloc[0]>(0.5+2*row.iloc[1]) else 'color:green' if row.iloc[0]<0.5-(2*row.iloc[1]) else '']
                                +['']
                                +['color: red' if row.iloc[2]>(0.5+2*row.iloc[3]) else 'color:green' if row.iloc[2]<0.5-(2*row.iloc[3]) else '']
                                +[''], axis=1)
                        .set_properties(**{'border': '1px black solid','padding':'8px'})
                        .set_table_attributes('style="border-collapse: collapse; border: 1px black solid;"')
                        .set_table_styles([{'selector': 'th', 'props': [('border', '1px solid black')]}]))
        with open("results/pattern_results.html", "w") as f:
            f.write("Confusion matrix of chart pattern signals and gru_gnn <br>")
            f.write(cm_signal_pattern.to_html())
            f.write("<br><br>")
            f.write("Accuracy of chart pattern signals and gru_gnn <br>")
            f.write(acc_compare_show.to_html())
        end_time = datetime.now().timestamp()
        print(f'Chart signal evaluation took {np.ceil((end_time - start_time) / 60)} minutes')

