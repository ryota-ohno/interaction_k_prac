import os
import pandas as pd
import time
import numpy as np
from src.make import exec_gjf
from src.utils import check_calc_status, invert_A, heri_to_A3
from src.optimize import get_params, get_init_para_csv
from src.listen import init_step,listen
from src.vdw import get_c_vec_vdw
import argparse

def init_process(args):
    
    auto_dir = args.auto_dir
    glide = args.glide
    heri = args.heri
    R1 = args.R1
    R2 = args.R2
    
    os.makedirs(os.path.join(auto_dir,'gaussian'), exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussview'), exist_ok=True)

    get_init_para_csv(auto_dir, R1, R2, heri, glide)
    
    auto_csv_path = os.path.join(auto_dir,'step2B_auto.csv')
    if not os.path.exists(auto_csv_path):        
        df_E_init = pd.DataFrame(columns = ['A1','A2','A3','E','E_p','E_t','a','b','cx','cy','cz','glide','machine_type','status'])
        df_E_init.to_csv(auto_csv_path,index=False)

    df_init=pd.read_csv(os.path.join(auto_dir,'step2B_init_params.csv'))
    df_init['status']='NotYet'
    df_init.to_csv(os.path.join(auto_dir,'step2B_init_params.csv'),index=False)
    
def main_process(args):
    
    auto_dir = args.auto_dir
    glide = args.glide
    heri = args.heri
    isInterlayer = False
    
    os.chdir(os.path.join(auto_dir,'gaussian'))
    
    auto_csv_path = os.path.join(auto_dir,'step2B_auto.csv')
    init_params_csv = os.path.join(auto_dir,'step2B_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    step = init_step(init_params_csv)
    isOver = False
    while not(isOver):
        #check
        isAvailable, machine_type = listen(auto_dir, heri,glide,isInterlayer)
        if not(isAvailable):
            time.sleep(1)
        else:
            df_cur=pd.read_csv(auto_csv_path)
            step,A1,A2,A3,a,b = get_params(auto_dir,df_cur,step)
            if A1==float('inf'):#全部inProgressな場合
                df_init_params = pd.read_csv(init_params_csv)
                if len(df_init_params[df_init_params['status']!='Done'])==0:
                    isOver=True
                continue
            print('step={}'.format(step))
            print('A1={},A2={},A3={},a={},b={}'.format(A1,A2,A3,a,b))

            alreadyCalculated = check_calc_status(df_cur,A1,A2,A3,a,b)
            if alreadyCalculated:
                continue
            else:
                A1_old,A2_old=invert_A(A1,A2)
                A3_old = heri_to_A3(A1_old,A2_old,heri)
                if isInterlayer:
                    cx,cy,cz = get_c_vec_vdw(A1_old,A2_old,A3_old,a,b,glide)
                else:
                    cx,cy,cz = [0.,0.,0.]
                exec_gjf(auto_dir,[A1_old,A2_old,A3_old,a,b,np.array([cx,cy,cz]),glide,isInterlayer], machine_type)
                df_newline = pd.Series([A1,A2,A3,0,0,0,a,b,cx,cy,cz,glide,machine_type,'InProgress'],index=df_cur.columns)
                df_cur=df_cur.append(df_newline,ignore_index=True)
                df_cur.to_csv(auto_csv_path,index=False)
            df_init_params = pd.read_csv(init_params_csv)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--init',action='store_true')
    parser.add_argument('--auto-dir',type=str,help='path to dir which includes gaussian, gaussview and csv')
    parser.add_argument('--R1', type=float, default=5.0)
    parser.add_argument('--R2', type=float, default=0.7)
    parser.add_argument('--heri', type=int, default=60)
    parser.add_argument('--glide', type=str, default='a')
    
    args = parser.parse_args()

    if args.init:
        print("----initial process----")
        init_process(args)
    
    print("----main process----")
    main_process(args)
    print("----finish process----")
    