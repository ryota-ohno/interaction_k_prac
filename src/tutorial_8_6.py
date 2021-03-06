import os
os.environ['HOME'] ='/home/ohno'
import pandas as pd
import time
import sys
from tqdm import tqdm
sys.path.append(os.path.join(os.environ['HOME'],'/Working/interaction_k_prac/'))
from make_8_2 import exec_gjf
from vdw_8_2 import vdw_R
from utils import get_E
import argparse
import numpy as np
from scipy import signal

def init_process(args):
    auto_dir = args.auto_dir##指定をするコマンドを受け取ったargsからautodirを受け取る
    monomer_name = args.monomer_name##monomer_nameについても同様
    order = 2
    os.makedirs(auto_dir, exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussian'), exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussview'), exist_ok=True)

    def get_init_para_csv(auto_dir,monomer_name):##初期値パラメータ決定
        init_params_csv = os.path.join(auto_dir, 'step1_init_params.csv')
        
        init_para_list = []
        A1 = 0; A2 = 0
        
        ################## TO BE FILLED (thetaの初期値) ##################
        theta_list = [round(i/2, 1) for i in range(50, 51)]
        #################################################################

        for theta in tqdm(theta_list):
            a_list = []; b_list = []; S_list = []
            a_clps=vdw_R(A1,A2,theta,0.0,'a',monomer_name)##slipped parallel a方向にどれだけずらさないといけないのか
            b_clps=vdw_R(A1,A2,theta,90.0,'b',monomer_name)##同様 b
            for theta_ab in range(0,91):
                R_clps=vdw_R(A1,A2,theta,theta_ab,'t',monomer_name)##t-shaped theta方向にどれだけずらさないといけないのか
                a=2*R_clps*np.cos(np.radians(theta_ab))##t-shaped基準でどれだけa方向にずらさないといけないのか
                b=2*R_clps*np.sin(np.radians(theta_ab))##b
                if (a_clps > a) or (b_clps > b):##slipped parallelの方がずらさないといけない
                    continue
                else:##a_clps < a and b_clps < b　　t-shapedの方がずらさないといけない
                    a = np.round(a,1);b = np.round(b,1)
                    a_list.append(a);b_list.append(b);S_list.append(a*b)
            local_minidx_list = signal.argrelmin(np.array(S_list), order=order)##sの極小値 接しないように離したうえでどれだけ近づけるか order:ノイズ除去
            if len(local_minidx_list[0])>0:
                for local_minidx in local_minidx_list[0]:
                    init_para_list.append([a_list[local_minidx],b_list[local_minidx],theta,'NotYet'])
            init_para_list.append([a_list[0],b_list[0],theta,'NotYet'])##最初は極小判定されない
            init_para_list.append([a_list[-1],b_list[-1],theta,'NotYet'])##最後も同様 list[-1]で最後を示す
            
        df_init_params = pd.DataFrame(np.array(init_para_list),columns = ['a','b','theta','status'])##dfはdateframe
        df_init_params.to_csv(init_params_csv,index=False)##step{n}_init_params.csvにa,b,θ,状態を書き込み
    
    get_init_para_csv(auto_dir,monomer_name)##auto-dir/step1_init_params.csvにmonomerにおける初期パラメータをいれる
    
    auto_csv_path = os.path.join(auto_dir,'step1.csv')
    if not os.path.exists(auto_csv_path):        
        df_E_init = pd.DataFrame(columns = ['a','b','theta','E','E_p1','E_p2','E_t','machine_type','status','file_name'])##step1.csvに列の名前を指定 E_p1:b E_p2:a
    else:
        df_E_init = pd.read_csv(auto_csv_path)
        df_E_init = df_E_init[df_E_init['status']!='InProgress']##statusがinprogressでない部分(終わってない部分)だけ抜き出す　statusはinprogressとnotyet,done
    df_E_init.to_csv(auto_csv_path,index=False)##上書き　行インデックスは削除

    df_init=pd.read_csv(os.path.join(auto_dir,'step1_init_params.csv'))
    df_init['status']='NotYet'
    df_init.to_csv(os.path.join(auto_dir,'step1_init_params.csv'),index=False)

def main_process(args):
    os.chdir(os.path.join(args.auto_dir,'gaussian'))##カレンとディレクトリをgaussianに変更
    isOver = False
    while not(isOver):##下のlistenでisover=Trueになるまで繰り返す
        #check
        isOver = listen(args)
        time.sleep(1)

def listen(args):
    auto_dir= args.auto_dir
    monomer_name = args.monomer_name
    num_nodes = args.num_nodes
    maxnum_machine2 = args.maxnum_machine2
    isTest = args.isTest

    auto_csv = os.path.join(auto_dir,'step1.csv')
    df_E = pd.read_csv(auto_csv)
    df_queue = df_E.loc[df_E['status']=='InProgress',['machine_type','file_name']]##行or列で要素を取り出し inprogress状態の行から2列の成分を取り出す
    machine_type_list = df_queue['machine_type'].values.tolist()##listに
    len_queue = len(df_queue)
    
    for idx,row in zip(df_queue.index,df_queue.values):##rowはdf_queueのvalue部分 上よりこれはmachine_typeとfilename
        machine_type,file_name = row##ここで引き渡し
        log_filepath = os.path.join(*[auto_dir,'gaussian',file_name])##*はlistの形で使うため
        if not(os.path.exists(log_filepath)):#logファイルが生成される直前だとまずいので
            continue##ループの最初に戻る
        E_list=get_E(log_filepath)
        if len(E_list)!=3:##len(Elist)が3でない時は3分子に関する計算が済んでいない
            continue
        else:
            len_queue-=1;machine_type_list.remove(machine_type)
            Et=float(E_list[0]);Ep1=float(E_list[1]);Ep2=float(E_list[2])##p1がb方向　p2がa方向
            E = 4*Et+2*(Ep1+Ep2)
            df_E.loc[idx, ['E_t','E_p1','E_p2','E','status']] = [Et,Ep1,Ep2,E,'Done']
            df_E.to_csv(auto_csv,index=False)##計算して得られたエネルギーとdone(計算終了)を書き込む
            break#2つ同時に計算終わったりしたらまずいので一個で切る
    isAvailable = len_queue < num_nodes 
    machine2IsFull = machine_type_list.count(2) >= maxnum_machine2##machine_type_list内の2の個数と比較
    machine_type = 1 if machine2IsFull else 2
    if isAvailable:
        params_dict = get_params_dict(auto_dir,num_nodes)
        if len(params_dict)!=0:#終わりがまだ見えないなら
            alreadyCalculated = check_calc_status(auto_dir,params_dict)
            if not(alreadyCalculated):
                file_name = exec_gjf(auto_dir, monomer_name, {**params_dict,'cx':0,'cy':0,'cz':0,'A1':0.,'A2':0.}, machine_type,isInterlayer=False,isTest=isTest)##上で取り込んだparamsdictからfile作成
                df_newline = pd.Series({**params_dict,'E':0.,'E_p1':0.,'E_p2':0.,'E_t':0.,'machine_type':machine_type,'status':'InProgress','file_name':file_name})##paramsdict : a,b,theta
                df_E=df_E.append(df_newline,ignore_index=True)
                df_E.to_csv(auto_csv,index=False)##step1.csvに書き込み
    
    init_params_csv=os.path.join(auto_dir, 'step1_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_init_params_done = filter_df(df_init_params,{'status':'Done'})##getparamsdictでinit_params_dict.csvにdoneを書き込める　filter_dfは下で定義
    isOver = True if len(df_init_params_done)==len(df_init_params) else False##その数が初期値の数と等しくなったらそれで終わりTrue ここでisOverがTrueになったら終わり
    return isOver

def check_calc_status(auto_dir,params_dict):
    df_E= pd.read_csv(os.path.join(auto_dir,'step1.csv'))
    if len(df_E)==0:
        return False
    df_E_filtered = filter_df(df_E, params_dict)##paramsdict と同じものだけ抜き取る
    df_E_filtered = df_E_filtered.reset_index(drop=True)##indexを振りなおす　保存しない
    try:
        status = get_values_from_df(df_E_filtered,0,'status')
        return status=='Done'
    except KeyError:
        return False

def get_params_dict(auto_dir, num_nodes):
    """
    前提:
        step1_init_params.csvとstep1.csvがauto_dirの下にある
    """
    init_params_csv=os.path.join(auto_dir, 'step1_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_cur = pd.read_csv(os.path.join(auto_dir, 'step1.csv'))
    df_init_params_inprogress = df_init_params[df_init_params['status']=='InProgress']##初期パラメータのうち計算中のもの
    fixed_param_keys = ['theta']##固定
    opt_param_keys = ['a','b']##最適化 この2つを+で足したものもリスト

    #最初の立ち上がり時
    if len(df_init_params_inprogress) < num_nodes:##?
        df_init_params_notyet = df_init_params[df_init_params['status']=='NotYet']##状態がnotyetのものを取り出す
        for index in df_init_params_notyet.index:
            df_init_params = update_value_in_df(df_init_params,index,'status','InProgress')##notyetをinprogressに
            df_init_params.to_csv(init_params_csv,index=False)
            params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys].to_dict()##params_dict(a,b,thetaとそれに対応するvalues)を作成し
            return params_dict##出力
    for index in df_init_params.index:##初期パラメータの表のある1行について
        df_init_params = pd.read_csv(init_params_csv)
        init_params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys].to_dict()##df.loc[index,]でdictを作成するindexを指定 a,b,thetaのdict
        fixed_params_dict = df_init_params.loc[index,fixed_param_keys].to_dict()##thetaのみのdict
        isDone, opt_params_dict = get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict)##get_opt_params_dictでTFとa,bを出力
        if isDone:##上でTrueが出た時
            # df_init_paramsのstatusをupdate
            df_init_params = update_value_in_df(df_init_params,index,'status','Done')##計算が終わったのでinitparams.csvのstatusをdoneに変更
            if np.max(df_init_params.index) < index+1:##init_params.csvの最後の行のとき　doneになる
                status = 'Done'
            else:##最後の行ではない時、次の行のものを借りてくる
                status = get_values_from_df(df_init_params,index+1,'status')
            df_init_params.to_csv(init_params_csv,index=False)##ここで更新
            
            if status=='NotYet':##is doneで計算がすんで次の行がまだ計算始まってない時                
                opt_params_dict = get_values_from_df(df_init_params,index+1,opt_param_keys)
                df_init_params = update_value_in_df(df_init_params,index+1,'status','InProgress')##つぎの行の状態をinprogressにしたものを出力　
                df_init_params.to_csv(init_params_csv,index=False)##更新
                return {**fixed_params_dict,**opt_params_dict}##この関数は(theta,a,b)を出力
            else:##次の行がすでに計算中なら
                continue##次のindexに

        else:##isdone =False
            df_inprogress = filter_df(df_cur, {**fixed_params_dict,**opt_params_dict,'status':'InProgress'})##dictで示される点かつinprogress(get optでfalseなら基本inprogress)
            print(df_inprogress)#############毎秒画面に表示されるのはこれ##########毎秒行われるlisten内にこのget_params_dictが存在
            if len(df_inprogress)>=1:
                print('continue')#######continueも基本的に毎秒表示
                continue
            return {**fixed_params_dict,**opt_params_dict}##この関数は(theta,a,b)を出力
    return {}##関数の終わり
        
def get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict):##df_curはstep1.csv　True(計算で安定構造に到達) or False(計算で掃き終わっていない) と(a,b)を出力
    df_val = filter_df(df_cur, fixed_params_dict)##特定のthetaの部分だけ抽出
    a_init_prev = init_params_dict['a']; b_init_prev = init_params_dict['b']; theta = init_params_dict['theta']##dictからvalueを抜き出し
    while True:##無限ループ
        E_list=[];ab_list=[]
        for a in [a_init_prev-0.1,a_init_prev,a_init_prev+0.1]:##aでのinit_params_dictの前後0.1
            for b in [b_init_prev-0.1,b_init_prev,b_init_prev+0.1]:##bでの
                a = np.round(a,1);b = np.round(b,1)##pythonの小数点を丸める
                df_val_ab = df_val[(df_val['a']==a)&(df_val['b']==b)&(df_val['theta']==theta)&(df_val['status']=='Done')]##df_valから指定した条件のものを抜き出し
                if len(df_val_ab)==0:##存在しない　エネルギーの計算が済んでいない
                    return False,{'a':a,'b':b}##ループを区切りまだ計算されていないa,bを出力
                ab_list.append([a,b]);E_list.append(df_val_ab['E'].values[0])##(a,b,),Eを抽出して加える
        a_init,b_init = ab_list[np.argmin(np.array(E_list))]##9点でEがminになるa,bを決定
        if a_init==a_init_prev and b_init==b_init_prev:
            return True,{'a':a_init,'b':b_init}
        else:
            a_init_prev=a_init;b_init_prev=b_init##_init_prevの更新 計算は済んでいないのでこの点を出力

def get_values_from_df(df,index,key):
    return df.loc[index,key]##そのまま dateframeからindexとkeyで場所を指定しそこでのvalueを出力

def update_value_in_df(df,index,key,value):##dateframeをindexとkeyで指定
    df.loc[index,key]=value##valueを代入
    return df

def filter_df(df, dict_filter):##このセクションで使われる関数 cict_filterはdict:key and value
    query = []
    for k, v in dict_filter.items():##key value
        if type(v)==str:##type 型　int str 等
            query.append('{} == "{}"'.format(k,v))
        else:
            query.append('{} == {}'.format(k,v))##上とこれでdict内の k==vの条件式を作る
    df_filtered = df.query(' and '.join(query))##それを接合し(join)複数のkに関する条件式を作る その条件に合うものだけを抜き取る(df.query)
    return df_filtered##dateframeを出力

if __name__ == '__main__':
    parser = argparse.ArgumentParser()##コマンドを受け取る　以下は受け取る引数
    
    parser.add_argument('--init',action='store_true')##--initを指定するとTrue しないとfalse
    parser.add_argument('--isTest',action='store_true')##--isTest以下同様
    parser.add_argument('--auto-dir',type=str,help='path to dir which includes gaussian, gaussview and csv')
    parser.add_argument('--monomer-name',default="demo", type=str,help='monomer name')
    parser.add_argument('--num-nodes',type=int,help='num nodes')
    parser.add_argument('--maxnum-machine2',type=int,help='num nodes')
    
    
    args = parser.parse_args()##これが下にあるようにinitやmain_processの引数になる

    if args.init:
        print("----initial process----")
        init_process(args)
    
    print("----main process----")
    main_process(args)
    print("----finish process----")
##ある(a,b)についてその点を含んだ9点について計算を行い最小の点が端ならそれについて同様に9点の計算をする(終わったものは計算しない)これで安定構造を探索していくプロセス   