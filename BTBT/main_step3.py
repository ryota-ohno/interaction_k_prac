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
    isInterlayer = True
    glide = args.glide
    heri = args.heri
    
    os.makedirs(os.path.join(auto_dir,'gaussian'), exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussview'), exist_ok=True)

    auto_csv_path = os.path.join(auto_dir,'step3.csv')
    df_cur=pd.read_csv(auto_csv_path)
    for A1,A2,a,b,glide in df_cur[['A1','A2','a','b','glide']]:
        machine_type = 2
        A1_old,A2_old=invert_A(A1,A2)
        A3_old = heri_to_A3(A1_old,A2_old,heri)
        cx,cy,cz = get_c_vec_vdw(A1_old,A2_old,A3_old,a,b,glide)
        exec_gjf(auto_dir,[A1_old,A2_old,A3_old,a,b,np.array([cx,cy,cz]),glide,isInterlayer], machine_type)
        df_newline = pd.Series([A1,A2,A3,0,0,0,a,b,cx,cy,cz,glide,machine_type,'InProgress'],index=df_cur.columns)
        df_cur=df_cur.append(df_newline,ignore_index=True)
    df_cur.to_csv(auto_csv_path,index=False)
            
def main_process(args):
    pass

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
    