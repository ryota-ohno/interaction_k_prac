import os
import pandas as pd
import time
import sys
from tqdm import tqdm
import argparse
import numpy as np
from scipy import signal
import scipy.spatial.distance as distance
import random

INTERACTION_PATH = os.path.join(os.environ['HOME'],'Working/interaction/')
sys.path.append(INTERACTION_PATH)

from src.make import exec_gjf
from src.vdw import vdw_R
from src.utils import get_E

def init_process(args):
    # 数理モデル的に自然な定義の元のparams initリスト: not yet
    # 結晶学的に自然なパラメータへ変換: not yet
    auto_dir = args.auto_dir
    order = 5
    monomer_name = args.monomer_name
    
    os.makedirs(os.path.join(auto_dir,'gaussian'), exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussview'), exist_ok=True)

    def get_init_para_csv(auto_dir,monomer_name):
        step1_params_csv = os.path.join(INTERACTION_PATH, '{}/step1/step1_min.csv'.format(monomer_name))
        init_params_csv = os.path.join(auto_dir, 'step2-twist_init_params.csv')
        
        init_para_list = []
#         A1_list =[0]; A2_list = [1,2,3,4,26,27,28,29,31,32,33,34,36,37,38,39]
        A1_list =[-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20]; A2_list = [32]
#         A1_list =[0,-5,-10]; A2_list = [0,5,10,15,20,25,30,35,40]
        
        df_step1 = pd.read_csv(step1_params_csv)
        
        a_,b_,theta = df_step1.loc[df_step1["E"].idxmin(),["a","b","theta"]].values
        step1_para_zip = [[b_,a_,90-theta]]
#         print(step1_para_zip)
#         d = 5.5
        
        for a_,b_,theta in step1_para_zip:
            print('a,b,theta')
            print(a_,b_,theta)
            for A1 in tqdm(A1_list):
                for A2 in A2_list:
                    if A1==0 and A2==0:
                        continue
                    a = a_# + 4 * d * abs(np.sin(np.radians(A1)))
                    b = b_/np.cos(np.radians(A2))
                    init_para_list.append([np.round(a,1),np.round(b,1),theta,A1,A2,'NotYet'])
                                
        df_init_params = pd.DataFrame(np.array(init_para_list),columns = ['a','b','theta','A1','A2','status'])
        df_init_params.to_csv(init_params_csv,index=False)
    
    get_init_para_csv(auto_dir,monomer_name)
    
    auto_csv_path = os.path.join(auto_dir,'step2-twist.csv')
    if not os.path.exists(auto_csv_path):        
        df_E = pd.DataFrame(columns = ['a','b','theta','A1','A2','E','E_p','E_t','machine_type','status','file_name'])
    else:
        df_E = pd.read_csv(auto_csv_path)
        df_E = df_E[df_E['status']!='InProgress']
    df_E.to_csv(auto_csv_path,index=False)

    df_init=pd.read_csv(os.path.join(auto_dir,'step2-twist_init_params.csv'))
    df_init['status']='NotYet'
    df_init.to_csv(os.path.join(auto_dir,'step2-twist_init_params.csv'),index=False)

def main_process(args):
    os.chdir(os.path.join(args.auto_dir,'gaussian'))
    isOver = False
    while not(isOver):
        #check
        isOver = listen(args)
        time.sleep(1)

def listen(args):
    auto_dir = args.auto_dir
    monomer_name = args.monomer_name
    num_nodes = args.num_nodes
    isTest = args.isTest
    fixed_param_keys = ['A1','A2']
    opt_param_keys = ['a','b','theta']

    auto_csv = os.path.join(auto_dir,'step2-twist.csv')
    df_E = pd.read_csv(auto_csv)
    df_queue = df_E.loc[df_E['status']=='InProgress',['machine_type','file_name']]
    machine_type_list = df_queue['machine_type'].values.tolist()
    len_queue = len(df_queue)
    maxnum_machine2 = 3#num_nodes/2 if num_nodes%2==0 else (num_nodes+1)/2
    
    for idx,row in zip(df_queue.index,df_queue.values):
        machine_type,file_name = row
        log_filepath = os.path.join(*[auto_dir,'gaussian',file_name])
        if not(os.path.exists(log_filepath)):#logファイルが生成される直前だとまずいので
            continue
        E_list=get_E(log_filepath)
        if len(E_list)!=2:
            continue
        else:
            len_queue-=1;machine_type_list.remove(machine_type)
            Et=float(E_list[0]);Ep=float(E_list[1])
            E = 4*Et+2*Ep
            df_E.loc[idx, ['E_t','E_p','E','status']] = [Et,Ep,E,'Done']
            df_E.to_csv(auto_csv,index=False)
            break#2つ同時に計算終わったりしたらまずいので一個で切る
    isAvailable = len_queue < num_nodes 
    machine2IsFull = machine_type_list.count(2) >= maxnum_machine2
    machine_type = 1 if machine2IsFull else 2
    if isAvailable:
        params_dict = get_params_dict(auto_dir,num_nodes, fixed_param_keys, opt_param_keys)
        if len(params_dict)!=0:#終わりがまだ見えないなら
            alreadyCalculated = check_calc_status(auto_dir,params_dict)
            if not(alreadyCalculated):
                file_name = exec_gjf(auto_dir, monomer_name, {**params_dict,'cx':0,'cy':0,'cz':0}, machine_type,isInterlayer=False,isTest=isTest)
                df_newline = pd.Series({**params_dict,'E':0.,'E_p':0.,'E_t':0.,'machine_type':machine_type,'status':'InProgress','file_name':file_name})
                df_E=df_E.append(df_newline,ignore_index=True)
                df_E.to_csv(auto_csv,index=False)
    
    init_params_csv=os.path.join(auto_dir, 'step2-twist_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_init_params_done = filter_df(df_init_params,{'status':'Done'})
    isOver = True if len(df_init_params_done)==len(df_init_params) else False
    return isOver

def check_calc_status(auto_dir,params_dict):
    df_E= pd.read_csv(os.path.join(auto_dir,'step2-twist.csv'))
    if len(df_E)==0:
        return False
    df_E_filtered = filter_df(df_E, params_dict)
    df_E_filtered = df_E_filtered.reset_index(drop=True)
    try:
        status = get_values_from_df(df_E_filtered,0,'status')
        return status=='Done'
    except KeyError:
        return False

def get_params_dict(auto_dir, num_nodes, fixed_param_keys, opt_param_keys):
    """
    前提:
        step2-twist_init_params.csvとstep2-twist.csvがauto_dirの下にある
    """
    init_params_csv=os.path.join(auto_dir, 'step2-twist_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_cur = pd.read_csv(os.path.join(auto_dir, 'step2-twist.csv'))
    df_init_params_inprogress = df_init_params[df_init_params['status']=='InProgress']
    
    #最初の立ち上がり時
    if len(df_init_params_inprogress) < num_nodes:
        df_init_params_notyet = df_init_params[df_init_params['status']=='NotYet']
        for index in df_init_params_notyet.index:
            df_init_params = update_value_in_df(df_init_params,index,'status','InProgress')
            df_init_params.to_csv(init_params_csv,index=False)
            params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys].to_dict()
            return params_dict
    for index in df_init_params.index:
        df_init_params = pd.read_csv(init_params_csv)
        init_params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys].to_dict()
        fixed_params_dict = df_init_params.loc[index,fixed_param_keys].to_dict()
        isDone, opt_params_dict = get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict)
        if isDone:
            # df_init_paramsのstatusをupdate
            df_init_params = update_value_in_df(df_init_params,index,'status','Done')
            if np.max(df_init_params.index) < index+1:
                status = 'Done'
            else:
                status = get_values_from_df(df_init_params,index+1,'status')
            df_init_params.to_csv(init_params_csv,index=False)
            
            if status=='NotYet':                
                opt_params_dict = get_values_from_df(df_init_params,index+1,opt_param_keys)
                df_init_params = update_value_in_df(df_init_params,index+1,'status','InProgress')
                df_init_params.to_csv(init_params_csv,index=False)
                return {**fixed_params_dict,**opt_params_dict}
            else:
                continue

        else:
            df_inprogress = filter_df(df_cur, {**fixed_params_dict,**opt_params_dict,'status':'InProgress'})
            if len(df_inprogress)>=1:
                continue
            return {**fixed_params_dict,**opt_params_dict}
    return {}
        
def get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict):
    df_val = filter_df(df_cur, fixed_params_dict)
    a_init_prev = init_params_dict['a']; b_init_prev = init_params_dict['b']; theta_init_prev = init_params_dict['theta']
    A1 = init_params_dict['A1']; A2 = init_params_dict['A2']
    
    while True:
        E_list=[];heri_list=[]
        for a in [a_init_prev-0.1,a_init_prev,a_init_prev+0.1]:
            for b in [b_init_prev-0.1,b_init_prev,b_init_prev+0.1]:
                a = np.round(a,1);b = np.round(b,1)
                for theta in [theta_init_prev-0.5,theta_init_prev,theta_init_prev+0.5]:
                    df_val_ab = df_val[
                        (df_val['a']==a)&(df_val['b']==b)&(df_val['theta']==theta)&
                        (df_val['A1']==A1)&(df_val['A2']==A2)&
                        (df_val['status']=='Done')
                                      ]
                    if len(df_val_ab)==0:
                        return False,{'a':a,'b':b,'theta':theta}
                    heri_list.append([a,b,theta]);E_list.append(df_val_ab['E'].values[0])
        a_init,b_init,theta_init = heri_list[np.argmin(np.array(E_list))]
        if a_init==a_init_prev and b_init==b_init_prev and theta_init==theta_init_prev:
            return True,{'a':a_init,'b':b_init, 'theta':theta_init}
        else:
            a_init_prev=a_init;b_init_prev=b_init;theta_init_prev=theta_init

def get_values_from_df(df,index,key):
    return df.loc[index,key]

def update_value_in_df(df,index,key,value):
    df.loc[index,key]=value
    return df

def filter_df(df, dict_filter):
    query = []
    for k, v in dict_filter.items():
        if type(v)==str:
            query.append('{} == "{}"'.format(k,v))
        else:
            query.append('{} == {}'.format(k,v))
    df_filtered = df.query(' and '.join(query))
    return df_filtered

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--init',action='store_true')
    parser.add_argument('--isTest',action='store_true')
    parser.add_argument('--auto-dir',type=str,help='path to dir which includes gaussian, gaussview and csv')
    parser.add_argument('--monomer-name',type=str,help='monomer name')
    parser.add_argument('--num-nodes',type=int,help='num nodes')
    
    args = parser.parse_args()

    if args.init:
        print("----initial process----")
        init_process(args)
    
    print("----main process----")
    main_process(args)
    print("----finish process----")
    