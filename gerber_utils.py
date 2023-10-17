import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_t
import os
import pickle
import plotly.express as px
import cvxpy as cp
from scipy.stats import norm


######## constants
# div_factor = {'W':5}


######## functions related to gerber implementation itself


def draw_tanhtanh_surface(a=1, b=1, c=0, d=0, rnge=5):

    # create a grid of x and y values
    x = np.linspace(-rnge, rnge, 1000)
    y = np.linspace(-rnge, rnge, 1000)
    X, Y = np.meshgrid(x, y)

    # compute the function values for each x and y pair
    Z = np.tanh(a*(X-c)) * np.tanh(b*(Y-d))

    # create a 3D plot of the function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X,Y)')
    
    if c != 0 or d != 0:
        ax.set_title(f'tanh({a}(x-{c})) * tanh({b}(y-{d}))')
    else:
        ax.set_title(f'tanh({a}x) * tanh({b}y)')
    plt.show()
    
    return

def draw_gerber_surface(rnge=5, **kwargs):

    # create a grid of x and y values
    if 'plot_granularity' in kwargs:
        x = np.linspace(-rnge, rnge, kwargs['plot_granularity'])
        y = np.linspace(-rnge, rnge, kwargs['plot_granularity'])
    else:
        x = np.linspace(-rnge, rnge, 300)
        y = np.linspace(-rnge, rnge, 300)
        
    X, Y = np.meshgrid(x, y)

    # compute the function values for each x and y pair
    Z = X.copy()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i][j] = gerber_weighting_m(X[i][j], Y[i][j], **kwargs)

    # create a 3D plot of the function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X,Y)')
    
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    else:
        ax.set_title(f'{kwargs["threshold_type"]}-threshold gerber')
    
    plt.show()
    
    return Z


def read_returns_snp_data():
    data = pd.read_parquet('snp500_data.parquet')

    df_close = data['Adj Close']
    df_close = df_close.dropna(axis=0, how='all')
    df_close = df_close.dropna(axis=1)
    df_returns = np.log(df_close).diff().dropna(axis=0, how='all') * 100
    
    return df_close, df_returns


def gerber_weighting_m(x, y, **kwargs):
    """
    one-threshold Gerber m_{ij} calculation (gerber 2021), or
    two-threshold Gerber m_{ij} calculation.
    """
    
    threshold_type = kwargs['threshold_type']
    
    if threshold_type.lower() == 'one':
        H_x = kwargs['H_x']
        H_y = kwargs['H_y']
        
        if x >= H_x and y >= H_y:
            return 1
        elif x <= -H_x and y <= -H_y:
            return 1
        elif x >= H_x and y <= -H_y:
            return -1
        elif x <= -H_x and y >= H_y:
            return -1
        else:
            return 0
        
    elif threshold_type.lower() == 'two':
        P = kwargs['P']
        Q = kwargs['Q']
        a = kwargs['a']
        
        if x >= Q and y >= Q:
            return 1
        elif x <= -Q and y <= -Q:
            return 1
        elif P <= x <= Q and P <= y <= Q:
            return 1
        elif -Q <= x <= -P and -Q <= y <= -P:
            return 1
        elif x >= Q and P <= y <= Q:
            return a
        elif x <= -Q and -Q <= y <= -P:
            return a
        elif P <= x < Q and y >= Q:
            return a
        elif -Q <= x <= -P and y <= -Q:
            return a
        elif x >= Q and -Q <= y <= -P:
            return -a
        elif x <= -Q and P <= y <= Q:
            return -a
        elif -Q <= x <= -P and y >= Q:
            return -a
        elif P <= x <= Q and y <= -Q:
            return -a
        elif x >= Q and y <= -Q:
            return -1
        elif x <= -Q and y >= Q:
            return -1
        elif P <= x < Q and -Q <= y <= -P:
            return -1
        elif -Q <= x <= -P and P <= y <= Q:
            return -1
#         elif np.abs(x) < P and np.abs(y) < Q:
#             return 0
        else:
            return 0
    
    elif threshold_type[:4].lower() == 'ring':
        C_list = kwargs['C_list']
        alpha_list = kwargs['alpha_list']
        
        if (np.abs(x) <= C_list[0]) or (np.abs(y) <= C_list[0]):
            return 0
        for i, c in enumerate(C_list[1:]):
            if (np.abs(x) <= c) and (np.abs(y) <= c):
                return alpha_list[i]
        else:
            return 1
        
    elif threshold_type.lower() == 'three':
        C_list = kwargs['C_list']
        alpha_list = kwargs['alpha_list']
        
        ############ positive weights
        ## corner conditions
        if (C_list[-1] <= x and C_list[-1] <= y) or (x < -C_list[-1] and y < -C_list[-1]):
            return alpha_list[0] # alpha_1=1
        elif (C_list[-1] <= x and C_list[-2] <= y < C_list[-1]) or (C_list[-1] <= -x and C_list[-2] <= -y < C_list[-1]):
            return alpha_list[1] # alpha_2
        elif (C_list[-2] <= x < C_list[-1] and C_list[-1] <= y) or (C_list[-2] <= -x < C_list[-1] and C_list[-1] <= -y):
            return alpha_list[1] #alpha_2
        elif (C_list[-1] <= x and C_list[-3] <= y < C_list[-2]) or (C_list[-1] <= -x and C_list[-3] <= -y < C_list[-2]):
            return alpha_list[2] #alpha3
        elif (C_list[-3] <= x < C_list[-2] and C_list[-1] <= y) or (C_list[-3] <= -x < C_list[-2] and C_list[-1] <= -y):
            return alpha_list[2] #alpha3
        else: # non-corner conditions
            for i, c in enumerate(C_list[:-1]):
                if (C_list[i] <= x < C_list[i+1] and C_list[i] <= y < C_list[i+1]) or (C_list[i] <= -x < C_list[i+1] and C_list[i] <= -y < C_list[i+1]):
                    return alpha_list[0]
            for i, c in enumerate(C_list[1:-1]):
                i += 1
                if (C_list[i] <= x < C_list[i+1] and C_list[i-1] <= y < C_list[i]) or (C_list[i] <= -x < C_list[i+1] and C_list[i-1] <= -y < C_list[i]):
                    return alpha_list[1]
                elif (C_list[i-1]<=x<C_list[i] and C_list[i]<=y<C_list[i+1]) or (C_list[i-1]<=-x<C_list[i] and C_list[i]<=-y<C_list[i+1]):
                    return alpha_list[1]

        ############ negative weights:
        ## corner conditions
        if (-x >= C_list[-1] and C_list[-1] <= y) or (x >= C_list[-1] and C_list[-1] <= -y):
            return -alpha_list[0] #-alpha1
        elif (x<-C_list[-1] and C_list[-2]<=y<C_list[-1]) or (-x<-C_list[-1] and C_list[-2]<=-y<C_list[-1]):
            return -alpha_list[1] #-alpha2
        elif (-C_list[-1]<=x<-C_list[-2] and y>C_list[-1]) or (-C_list[-1]<=-x<-C_list[-2] and -y>C_list[-1]):
            return -alpha_list[1] #-alpha2
        elif (C_list[-1]<=x and -C_list[-2]<=y<-C_list[-3]) or (C_list[-1]<=-x and -C_list[-2]<=-y<-C_list[-3]):
            return -alpha_list[2] #-alpha3
        elif (-C_list[-2]<=x<-C_list[-3] and C_list[-1]<=y) or (-C_list[-2]<=-x<-C_list[-3] and C_list[-1]<=-y):
            return -alpha_list[2] #-alpha3
        else:
            for i, c in enumerate(C_list[:-1]):
                if (C_list[i] <= -x < C_list[i+1] and C_list[i] <= y < C_list[i+1]) or (C_list[i] <= x < C_list[i+1] and C_list[i] <= -y < C_list[i+1]):
                    return -alpha_list[0] #-alpha1
            for i, c in enumerate(C_list[1:-1]):
                i += 1
                if (-C_list[i+1] <= x < -C_list[i] and C_list[i-1] <= y < C_list[i]) or (-C_list[i+1] <= -x < -C_list[i] and C_list[i-1] <= -y < C_list[i]):
                    return -alpha_list[1] #-alpha2
                elif (C_list[i-1]<=x<C_list[i] and -C_list[i+1]<=y<-C_list[i]) or (C_list[i-1]<=-x<C_list[i] and -C_list[i+1]<=-y<-C_list[i]):
                    return -alpha_list[1] #-alpha2
        return 0 # otherwise
        

def tanh_weighting_function(x, y, a=1, b=1, c=0, d=0):
    """
    computes tanh(a(x-c))tanh(b(y-d))
    """
    return np.tanh(a*(x-c)) * np.tanh(b*(y-d))

def is_pos_semi_def(mat, tol=None):
    """
    checks matrix is positive semi-definite.
    tolerance formula from https://www.mathworks.com/help/matlab/math/determine-whether-matrix-is-positive-definite.html
    """
    eig = np.linalg.eigvals(mat)
    if tol is None:
        tol = len(eig) * np.spacing(eig.max().real)
    return np.all(eig >= -tol)


def tanhtanh_two_series(df, a=1, b=1, c=0, d=0):
    df = pd.DataFrame(np.array(df)).copy()
    df = df.apply(lambda row:tanh_weighting_function(row[1], row[0], a=a, b=b, c=c, d=d), axis=1)
    return df

def gerber_weights_two_series(df, **kwargs):
    # one-threshold or two-treshold gerber
    df = pd.DataFrame(np.array(df)).copy()
    df = df.apply(lambda row:gerber_weighting_m(row[1], row[0], **kwargs), axis=1)
    return df

def fill_in_zero_in_symmetric_matrix(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i,j] == 0:
                mat[i,j] = mat[j,i]
    return mat

def fill_nan_diagonals(mat):
    mat = mat.copy()
    # replace nan diagonal with 1. Can have nan in diagonal if denominator is 0. i.e. no N_nn = T for that ij
    for i in range(mat.shape[0]):
        if np.isnan(mat.iloc[i,i]):
            mat.iloc[i,i] = 1
    return mat

def calc_num_den_matrix(obs, threshold_type, **kwargs):
    if threshold_type == 'one':
        threshold = np.array(kwargs['threshold'])
        
        U = (obs >= threshold).astype(int)
        D = (obs <= -threshold).astype(int)
        nn = (np.abs(obs) < threshold).astype(int)

        N_uu = U.T @ U
        N_dd = D.T @ D
        N_nn = nn.T @ nn

        N_conc = N_uu + N_dd
        N_disc = U.T @ D + D.T @ U

        numerator = N_conc - N_disc

        nn = (np.abs(obs) < threshold).astype(int)
        N_nn = nn.T @ nn
        denominator = obs.shape[0] - N_nn
        
        return {'numerator':numerator,
                'denominator':denominator}
    
    elif threshold_type == 'two':
        P = np.array(kwargs['P'])
        Q = np.array(kwargs['Q'])
        alpha = kwargs['alpha']
        
        U1 = (obs >= Q).astype(int)
        U2 = ((P <= obs) & (obs < Q)).astype(int)
        D1 = (obs < -Q).astype(int)
        D2 = ((-Q <= obs) & (obs < -P)).astype(int)
        
        N_u = U1.T @ U1 + U2.T @ U2 + alpha*(U1.T @ U2 + U2.T @ U1)
        N_d = D1.T @ D1 + D2.T @ D2 + alpha*(D1.T @ D2 + D2.T @ D1)
        
        N_conc = N_u + N_d
        N_disc = U1.T @ D1 + U2.T @ D2 + D1.T @ U1 + D2.T @ U2 + alpha*(U1.T @ D2 + U2.T @ D1 + D1.T @ U2 + D2.T @ U1)
        
        nn = (np.abs(obs) < P).astype(int)    
        N_nn = nn.T @ nn
        
        numerator = N_conc - N_disc
        denominator = U1.shape[0] - N_nn
        
        return {'numerator':numerator,
                'denominator':denominator}
    
    elif threshold_type == 'three':
        C_list = kwargs['C_list']
        alpha_list = kwargs['alpha_list']
        
        U1 = ((C_list[0] <= obs) & (obs < C_list[1])).astype(int)
        U2 = ((C_list[1] <= obs) & (obs < C_list[2])).astype(int)
        U3 = (C_list[2] <= obs).astype(int)
        D1 = ((C_list[0] <= -obs) & (-obs < C_list[1])).astype(int)
        D2 = ((C_list[1] <= -obs) & (-obs < C_list[2])).astype(int)
        D3 = (C_list[2] <= -obs).astype(int)
        nn = (np.abs(obs) < C_list[0]).astype(int)   
        
        N_conc = alpha_list[0] * (U1.T@U1 + U2.T@U2 + U3.T@U3 + D1.T@D1 + D2.T@D2 + D3.T@D3) \
               + alpha_list[1] * (U1.T@U2 + U2.T@U1 + U3.T@U2 + U2.T@U3 + D1.T@D2 + D2.T@D1 + D3.T@D2 + D2.T@D3) \
               + alpha_list[2] * (U3.T@U1 + U1.T@U3 + D3.T@D1 + D1.T@D3)
        N_disc = alpha_list[0] * (U1.T@D1 + U2.T@D2 + U3.T@D3 + D1.T@U1 + D2.T@U2 + D3.T@U3) \
               + alpha_list[1] * (U1.T@D2 + U2.T@D1 + U3.T@D2 + U2.T@D3 + D1.T@U2 + D2.T@U1 + D3.T@U2 + D2.T@U3) \
               + alpha_list[2] * (U3.T@D1 + U1.T@D3 + D3.T@U1 + D1.T@U3)
        N_nn = nn.T @ nn
        
        numerator = N_conc - N_disc
        denominator = U1.shape[0] - N_nn

        return {'numerator':numerator,
                'denominator':denominator}

def calc_gerber_corr_matrix(obs, denominator_version=2, return_all=True, **kwargs):
    """
    compute 1 or 2-threshold gerber statistic depending on kwargs['threshold_type']
    """
    
    threshold_type = kwargs['threshold_type']
    
    if threshold_type == 'one':
        threshold = np.array(kwargs['threshold'])
        
        U = (obs >= threshold).astype(int)
        D = (obs <= -threshold).astype(int)
        nn = (np.abs(obs) < threshold).astype(int)

        N_uu = U.T @ U
        N_dd = D.T @ D
        N_nn = nn.T @ nn

        N_conc = N_uu + N_dd
        N_disc = U.T @ D + D.T @ U

        numerator = N_conc - N_disc
        denominator = U.shape[0] - N_nn # T - n_ij^nn, # eq 11 in gerber 2021

        if denominator_version == 1:
            # version = 1 means eq 4 in gerber 2021
            denominator = N_conc + N_disc # (eq 4 in gerber 2021).
            # G = (N_conc - N_disc) / (N_conc + N_disc) # sometimes not psd
        G = numerator / denominator
        
        # replace nan diagonal with 1. Can have nan in diagonal if denominator is 0. i.e. no N_nn = T for that ij
        G = fill_nan_diagonals(G)
        
        if return_all:
            return G, numerator, denominator
        
        return G, 'placeholder', 'placeholder'
    
    elif threshold_type == 'two':
        P = np.array(kwargs['P'])
        Q = np.array(kwargs['Q'])
        alpha = kwargs['alpha']
        
        U1 = (obs >= Q).astype(int)
        U2 = ((P <= obs) & (obs < Q)).astype(int)
        D1 = (obs < -Q).astype(int)
        D2 = ((-Q <= obs) & (obs < -P)).astype(int)
        
        N_u = U1.T @ U1 + U2.T @ U2 + alpha*(U1.T @ U2 + U2.T @ U1)
        N_d = D1.T @ D1 + D2.T @ D2 + alpha*(D1.T @ D2 + D2.T @ D1)
        
        N_conc = N_u + N_d
        N_disc = U1.T @ D1 + U2.T @ D2 + D1.T @ U1 + D2.T @ U2 + alpha*(U1.T @ D2 + U2.T @ D1 + D1.T @ U2 + D2.T @ U1)
        
#         G = (N_conc - N_disc) / (N_conc + N_disc) # not psd when d=10
        nn = (np.abs(obs) < P).astype(int)    
        N_nn = nn.T @ nn
        
        numerator = N_conc - N_disc
        denominator = U1.shape[0] - N_nn # eq 11 in gerber 2021
        
        if denominator_version == 1:
            # version = 1 means eq 4 in gerber 2021
            denominator = N_conc + N_disc # (eq 4 in gerber 2021).
        
        G = numerator / denominator 
        
        # replace nan diagonal with 1. Can have nan in diagonal if denominator is 0. i.e. no N_nn = T for that ij
        G = fill_nan_diagonals(G)
        
        if return_all:
            return G, numerator, denominator
        
        return G, 'placeholder', 'placeholder'
    
    elif threshold_type == 'three':
        C_list = kwargs['C_list']
        alpha_list = kwargs['alpha_list'] #alpha_list includes alpha_1 (for diagonals, usually equal to 1)
        
        assert (len(C_list)==3 and len(alpha_list)==3)
        
        U1 = ((C_list[0] <= obs) & (obs < C_list[1])).astype(int)
        U2 = ((C_list[1] <= obs) & (obs < C_list[2])).astype(int)
        U3 = (C_list[2] <= obs).astype(int)
        D1 = ((C_list[0] <= -obs) & (-obs < C_list[1])).astype(int)
        D2 = ((C_list[1] <= -obs) & (-obs < C_list[2])).astype(int)
        D3 = (C_list[2] <= -obs).astype(int)
        nn = (np.abs(obs) < C_list[0]).astype(int)   
        
        N_conc = alpha_list[0] * (U1.T@U1 + U2.T@U2 + U3.T@U3 + D1.T@D1 + D2.T@D2 + D3.T@D3) \
               + alpha_list[1] * (U1.T@U2 + U2.T@U1 + U3.T@U2 + U2.T@U3 + D1.T@D2 + D2.T@D1 + D3.T@D2 + D2.T@D3) \
               + alpha_list[2] * (U3.T@U1 + U1.T@U3 + D3.T@D1 + D1.T@D3)
        N_disc = alpha_list[0] * (U1.T@D1 + U2.T@D2 + U3.T@D3 + D1.T@U1 + D2.T@U2 + D3.T@U3) \
               + alpha_list[1] * (U1.T@D2 + U2.T@D1 + U3.T@D2 + U2.T@D3 + D1.T@U2 + D2.T@U1 + D3.T@U2 + D2.T@U3) \
               + alpha_list[2] * (U3.T@D1 + U1.T@D3 + D3.T@U1 + D1.T@U3)
        N_nn = nn.T @ nn
        
        numerator = N_conc - N_disc
        denominator = U1.shape[0] - N_nn
        if denominator_version == 1:
            denominator = N_conc + N_disc # (eq 4 in gerber 2021)
        
        G = numerator / denominator
        G = fill_nan_diagonals(G)
        
        if return_all:
            return G, numerator, denominator
        
        return G, 'placeholder', 'placeholder'
    
    elif 'ring' in threshold_type:
        C_list = kwargs['C_list']
        alpha_list = kwargs['alpha_list'] # alpha_list might have different ordering to threshold_type==three. DOUBLE CHECK.
        
        U1 = ((C_list[0] <= np.abs(obs)) & (np.abs(obs) < C_list[1]) ).astype(int)
        U2 = (C_list[1] <= np.abs(obs)).astype(int)

        nn = (np.abs(obs) < C_list[0]).astype(int)    
        N_nn = nn.T @ nn 

        numerator = U2.T @ U2 + U1.T @ U2 + U2.T @ U1 + alpha_list[0] * U1.T @ U1 
        denominator = U1.shape[0] - N_nn - (1-alpha_list[0]) * U1.T @ U1
        G = numerator / denominator 
        G = fill_nan_diagonals(G)
        
        if return_all:
            return G, numerator, denominator
        return G, 'placeholder', 'placeholder'
    
    elif 'tanh-tanh' in threshold_type:
        a = kwargs['a']
        b = kwargs['b']
        c = kwargs['c']
        d = kwargs['d']
        
        K = obs.shape[1] # no. of stocks
        
        M_sum = np.zeros(K*K).reshape(K,K)
        M_abs_sum = np.zeros(K*K).reshape(K,K) # matrix consisting of denominator entries
        
        triu_indices = np.triu_indices(K)
        for i,j in zip(triu_indices[0], triu_indices[1]):

            # create M_sum and M_abs_sum matrices for upper triangular indicies
            M_sum[i,j] = tanhtanh_two_series(obs.iloc[:,[i,j]], a=a, b=b, c=c, d=d).values.sum()
            M_abs_sum[i,j] = np.abs(tanhtanh_two_series(obs.iloc[:,[i,j]], a=a, b=b, c=c, d=d).values).sum()

            # fill in lower triangular parts by looking at symmetric entries
            M_sum = fill_in_zero_in_symmetric_matrix(M_sum)
            M_abs_sum = fill_in_zero_in_symmetric_matrix(M_abs_sum)

        # create gerber corr (eq 3 in 2021 paper) matrix. (they say sometimes this is not PSD)
        G = M_sum / M_abs_sum
        
        return G, M_sum, M_abs_sum
        
        
    
    else:
        print(f'unidentified threshold_type: {threshold_type}')
        return
    
def normalize_columns(df):
    """
    used to normalize observations/simulations of log returns.
    """
    return (df-df.mean())/df.std()


def is_valid_alphas(alpha_list, threshold_type='three'):
    """
    Check whether input alhpa_list conforms to PSD conditions of alpha_list for each threshold_type of Gerber corr
    """
    if threshold_type == 'three':
        alpha1 = alpha_list[0]
        alpha2 = alpha_list[1]
        alpha3 = alpha_list[2]
        
        return (np.abs(alpha2) < 1) and (alpha3 > 2*alpha2**2-1) and (alpha3 < 1)
    
##### portfolio optimization related functions


def create_dir(fpath):
    if not os.path.exists(fpath):
        os.makedirs(fpath)



def pickle_obj(obj, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def unpickle_obj(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
        return b
    


def plot_ts(style, df, key, y_lab):
    if style.lower() == 'plotly':
        fig = px.line(df,
                    labels={"index":"time",
                            "value":y_lab})

        fig.update_layout(
            title=key)

        fig.show()

    elif style.lower() == 'matplotlib':
        for col in df.columns:
            plt.plot(df[col], label=col)
        plt.title(key)
        plt.ylabel(y_lab)
        plt.legend()
        plt.show()
    return

def formulate_problem(mu, sigma, target_annual_vol, resample_period, prev_weights, phi):
    """
    formulate convex portfolio optimization problem in cvxpy
    target_annual_vol: 0.16 for example for 16% annualized vol 
    """

    if resample_period == 'd':
        annual_factor = 252
    elif resample_period == 'W':
        annual_factor = 52
    elif resample_period == 'M':
        annual_factor = 12
    else:
        raise

    # Defining initial variables
    x = cp.Variable((mu.shape[1], 1))
    daily_vol_target = target_annual_vol/(annual_factor**0.5) # daily max sd. allowed in constraint
    ret = mu @ x

    # Budget and weights constraints
    constraints = [cp.sum(x) == 1,
                x <= 1,
                x >= 0]

    # Defining risk constraint and objective
    risk = cp.quad_form(x, sigma)
    # daily variance constraint
    constraints += [risk <= daily_vol_target**2] 
    trans_cost = 0 if prev_weights is None else phi * cp.sum(cp.abs(x - prev_weights.to_numpy()))
    objective = cp.Maximize(ret - trans_cost)

    prob = cp.Problem(objective, constraints)

    return prob, objective, constraints, x, daily_vol_target, daily_vol_target**2

def plot_ts(style, df, key, y_lab):
    """
    ts plotting function to be used in a for loop (not using subplots), one by one
    """
    if style.lower() == 'plotly':
        fig = px.line(df,
                    labels={"index":"time",
                            "value":y_lab})

        fig.update_layout(
            title=key)

        fig.show()

    elif style.lower() == 'matplotlib':
        for col in df.columns:
            plt.plot(df[col], label=col)
        plt.title(key)
        plt.ylabel(y_lab)
        plt.legend()
        plt.show()
    return


def calc_annual_returns(df, resample_period, do_round=None):
    if resample_period.upper() == 'M':
        factor = 12
    elif resample_period.upper() == 'W':
        factor = 52
    elif resample_period.upper() == 'D':
        factor = 252
    else:
        assert (False)

    num_rows = df.shape[0]
    num_compound = num_rows/factor

    if do_round:
        return ((df+1).cumprod().iloc[-1]**(1/num_compound) - 1).round(do_round)

    # return geometric annualized returns
    return (df+1).cumprod().iloc[-1]**(1/num_compound) - 1


def get_min_max_range(df_returns_portfolio_dict, nrows, ncols):
    """
    get min and max y ranges for fixed lookback, for plotting for various lookback-variance combinations
    """
    mins = []
    maxs = []

    keys = list(df_returns_portfolio_dict.keys())


    for i in range(nrows):
        curr_min = 9999
        curr_max = -9999        

        for j in range(ncols):
            cum_prod = (df_returns_portfolio_dict[keys[i*ncols+j]]+1).cumprod()

            curr_min = cum_prod.min().min() if cum_prod.min().min() < curr_min else curr_min
            curr_max = cum_prod.max().max() if cum_prod.max().max() > curr_max else curr_max
            
        mins.append(curr_min-0.1)
        maxs.append(curr_max+0.1)

    return mins, maxs
    
    
def plot_cum_returns(df_dict, backtest_name, nrows, ncols, resample_period, do_round, figsize=(40,40)):
    """
    plot cum returns.
    Each row is lookback
    each col is variance target.
    Displays annualized return for each cov type.
    """

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    keys = list(df_dict.keys())

    mins, maxs = get_min_max_range(df_dict, nrows, ncols)

    for i, row in enumerate(axes):
        for j, col in enumerate(row):
            ax = axes[i,j]

            annual_returns = {}
            
            for col_name in df_dict[keys[i*ncols+j]].columns:
                ax.plot((df_dict[keys[i*ncols+j]][col_name]+1).cumprod(), label=col_name)
                ax.set_ylim([mins[i], maxs[i]])

                annual_returns[col_name] = calc_annual_returns(df_dict[keys[i*ncols+j]][col_name], resample_period, do_round=do_round)

            if i == 0 and j==0:
                ax.legend(loc='upper left', ncol=1)
            
            stats = ''
            for key, val in annual_returns.items():
                stats += f'{key}_annual: {(val*100).round(4)}%'
                stats += '\n'

            ax.set_title(f'{keys[i*ncols+j]}\n{stats}')

    plt.suptitle(backtest_name, fontsize=18, y=1)
    fig.tight_layout()
    plt.show()


def plot_returns_hist(df, filename, cov_types_to_test, quantile_values=[0.01, 0.05], figsize=(10,12)):
    """
    plot portfolio returns histogram with statistics displayed
    """
    
    nrows = len(cov_types_to_test)
    ncols = 1
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    min_ret = df[filename].min().min()
    max_ret = df[filename].max().max()

    for i, return_type in enumerate([cov_type + '_return' for cov_type in cov_types_to_test]):

        ax = axes[i]
        df_plot = df[filename][return_type]

        h, edges, _ = ax.hist(df_plot, alpha = 0.5, density = False, bins = 100)
        ax.set_title(return_type)

        param = norm.fit(df_plot.dropna())   # Fit a normal distribution to the data
        x = np.linspace(min_ret, max_ret, 1000) # x-values
        binwidth = np.diff(edges).mean()
        ax.plot(x, norm.pdf(x, *param)*h.sum()*binwidth, color='r', label='ePdf')

        ax.axvline(df_plot.mean(), color='blue', label=f'mean: {round((df_plot.mean()*100),4)}%')
        if quantile_values:
            for q in quantile_values:
                ax.axvline(norm.ppf(q, *param), color='orange', label=f'{int((1-q)*100)}% VaR: {round((norm.ppf(q, *param)*100),4)}%')
            for q in quantile_values:
                q = 1-q
                ax.axvline(norm.ppf(q, *param), color='green', label=f'{int((1-q)*100)}% VaR: {round((norm.ppf(q, *param)*100),4)}%')
        ax.legend()


    fig.tight_layout()
    plt.suptitle(filename, fontsize=18, y=1.01)
    

def calc_VaR(df, q):
    """
    fit normal distribution to data and find quantile
    """
    param = norm.fit(df.dropna())
    return norm.ppf(q, *param)

def get_annual_factor(resample_period):
    """
    for annualizing high-freq data to yearly rate
    """
    if resample_period == 'd':
        annual_factor = 252
    elif resample_period == 'W':
        annual_factor = 52
    elif resample_period == 'M':
        annual_factor = 12
    else:
        raise
    return annual_factor


def plot_frontier(df, title, plot_theme='default', **kwargs):
    """
    plot efficient frontier
    """

    if plot_theme == 'default':
        try:
            value_vars = kwargs['value_vars']
        except:
            value_vars=['historical', '1-t gerber', '2-t gerber']
        # plotting return vs annual var for cov types, across lookback periods
        df = df.T.reset_index().rename(columns={'index':'annualized_variance'})
        df = df.melt(id_vars=['annualized_variance'], value_vars=value_vars, var_name='method', value_name='annualized_return')
        fig = px.line(df, x='annualized_variance', y='annualized_return', color='method', markers=True)

    elif plot_theme == 'vary alpha_2t':
        alpha_list = kwargs["alpha_list"]
        df = df.T.reset_index().rename(columns={'index':'annualized_variance'})
        df = df.melt(id_vars='annualized_variance', value_vars=alpha_list, var_name='alpha', value_name='annualized_return')
        fig = px.line(df, x='annualized_variance', y='annualized_return', color='alpha', markers=True)

    elif plot_theme == 'shift_PQ':
        df = df.T.reset_index().rename(columns={'index':'annualized_variance'})
        df.columns = [str(col) for col in df.columns]
        df = df.melt(id_vars='annualized_variance', value_vars=[str(col) for col in df.columns][1:], var_name='[P,Q]', value_name='annualized_return')
        fig = px.line(df, x='annualized_variance', y='annualized_return', color='[P,Q]', markers=True)

    elif plot_theme == 'vary_Q':
        Q_list = kwargs["Q_list"]
        df = df.T.reset_index().rename(columns={'index':'annualized_variance'})
        df = df.melt(id_vars='annualized_variance', value_vars=Q_list, var_name='Q', value_name='annualized_return')
        fig = px.line(df, x='annualized_variance', y='annualized_return', color='Q', markers=True)

    elif plot_theme == 'vary lmbda':
        lmbda_list = kwargs["lmbda_list"]
        df = df.T.reset_index().rename(columns={'index':'annualized_variance'})
        df = df.melt(id_vars='annualized_variance', value_vars=lmbda_list, var_name='lmbda', value_name='annualized_return')
        fig = px.line(df, x='annualized_variance', y='annualized_return', color='lmbda', markers=True)
    
    else:
        raise

    fig.update_layout(
        width=600,
        height=400,
        title=title
    )

    fig.update_traces(
        marker_size=10
    )

    fig.show()


def get_backtest_name_params(backtest_name, resample_period, phi, gerber_1t_threshold, gerber_2t_thresholds, alpha_2t, data_name):
    return f'{backtest_name}_{data_name}_{resample_period}_phi{phi}_1tT{gerber_1t_threshold}_2tT{gerber_2t_thresholds}_alpha2T{alpha_2t}'

def get_backtest_name_params_ewma(backtest_name, resample_period, phi, gerber_1t_threshold, gerber_2t_thresholds, alpha_2t, data_name, lmbda):
    return f'{backtest_name}_{data_name}_lmbda{lmbda}_1tT{gerber_1t_threshold}_2tT{gerber_2t_thresholds}_alpha2T{alpha_2t}'

def get_lookback_list(resample_period):
    if resample_period.upper() == 'M':
        return [6, 9, 12]
    elif resample_period.upper() == 'W':
        return [4, 6, 12]
    elif resample_period.upper() == 'D':
        return [30, 90, 180, 360]
    else:
        raise

def get_annual_var_constraint_list():
    return [0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4]

def row_select(df, backtest_identifier, date, data_name):
    df_select = df[(df['date'] == date) & (df['backtest_name'])]
    df_select = df_select[df_select['backtest_name'].str.contains(backtest_identifier)]

    if data_name == 'NaN':
        return df_select

    df_select = df_select[df_select['data_name'].str.contains(data_name)]
    return df_select


def halflife_to_lmbda(halflife):
    return 2**(-1/halflife)

def lmbda_to_halflife(lmbda):
    return -np.log(2)/np.log(lmbda)

def find_min_ramp_up(obs, threshold_type, **kwargs):
    """
    find min possible q for calculating initial ramp up period
    """
    if threshold_type == 'one':
        threshold = kwargs['threshold']
    
        for ramp_up in range(len(obs)):
            den = calc_num_den_matrix(obs.iloc[:ramp_up], threshold_type=threshold_type, threshold=threshold)['denominator']
            zero_count = (den==0).sum().sum()
            if zero_count == 0:
                break
    
    elif threshold_type == 'two':
        P = np.array(kwargs['P'])
        Q = np.array(kwargs['Q'])
        alpha = kwargs['alpha']
        
        for ramp_up in range(len(obs)):
            den = calc_num_den_matrix(obs.iloc[:ramp_up], threshold_type=threshold_type, 
                                                  P=P,
                                                Q=Q,
                                                alpha=alpha)['denominator']
            zero_count = (den==0).sum().sum()
            if zero_count == 0:
                break
            
    return ramp_up

def calc_G_tilde(obs, t, threshold_type, **kwargs):
    """
    calculate \tilde{G} which uses minimum possible q to define non-zero denominator entries.
    G tilde is used in EWMA calculation as the 'current' time t data.
    """
    
    if threshold_type == 'one':
        threshold = kwargs['threshold']
        
        num_den = calc_num_den_matrix(obs.iloc[[t]], threshold_type=threshold_type, threshold=threshold)
        num = num_den['numerator']
        den = num_den['denominator']
        
        
        if den.min().min() <= 0:
            for q in range(1,t):
                num_den = calc_num_den_matrix(obs.iloc[t-q:t+1], threshold_type=threshold_type, threshold=threshold)
                num = num_den['numerator']
                den = num_den['denominator']
                if den.min().min() > 0:
#                     print(den)
                    break
        else:
            q=0
            num_den = calc_num_den_matrix(obs.iloc[t-q:t+1], threshold_type=threshold_type, threshold=threshold)
            num = num_den['numerator']
            den = num_den['denominator']
            
    elif threshold_type == 'two':
        P = np.array(kwargs['P'])
        Q = np.array(kwargs['Q'])
        alpha = kwargs['alpha']
        
        num_den = calc_num_den_matrix(obs.iloc[[t]], threshold_type=threshold_type, P=P, Q=Q, alpha=alpha)
        num = num_den['numerator']
        den = num_den['denominator']

        if den.min().min() <= 0:
            for q in range(1,t):
                num_den = calc_num_den_matrix(obs.iloc[t-q:t+1], threshold_type=threshold_type, P=P, Q=Q, alpha=alpha)
                num = num_den['numerator']
                den = num_den['denominator']
                if den.min().min() > 0:
#                     print(den)
                    break
        else:
            q=0
            num_den = calc_num_den_matrix(obs.iloc[t-q:t+1], threshold_type=threshold_type, P=P, Q=Q, alpha=alpha)
            num = num_den['numerator']
            den = num_den['denominator']
    return {'corr':num/den, 'num':num, 'den':den, 'q':q}


def backtest_stats_to_latex(df, freq_lookback):
    df_disp = df[freq_lookback].T.copy().round(4) * 100

    df_disp = df_disp.rename(columns={'historical':'Historical',
                             '1-t gerber':'1-T Gerber',
                             '2-t gerber':'2-T Gerber'},
                    index={f'{freq_lookback}_var0.05':'5%',
                           f'{freq_lookback}_var0.1':'10%',
                           f'{freq_lookback}_var0.15':'15%',
                           f'{freq_lookback}_var0.2':'20%',
                           f'{freq_lookback}_var0.25':'25%',
                           f'{freq_lookback}_var0.3':'30%',
                           f'{freq_lookback}_var0.35':'35%',
                           f'{freq_lookback}_var0.4':'40%',
                           'annual_returns': 'Annual Returns (%)',
                           'cumm_returns': 'Cumulative Returns (%)',
                           '99%_VaR': '99% VaR',
                           '95%_VaR': '95% VaR',
                           'sharpe_ratio': 'Sharpe Ratio'})
    
    for var in np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])*100:
        var = int(var)
        df_disp.loc[(f'{var}%', 'Sharpe Ratio')] = df_disp.loc[(f'{var}%', 'Sharpe Ratio')] / 100
        
    return df_disp, df_disp.to_latex()