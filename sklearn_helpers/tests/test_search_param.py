from sklearn_helpers import search_param

import math
import numpy as np
import pandas as pd

def f(x,y):
    return x**2+y**2



def test_fl_search():
    params={'x':list(range(-10,10)),'y':list(range(-10,10))}
    real_param,real_tracking=search_param.fl_search(f,params=params,n_iter=30)
    expected_param={'x':0,'y':0}
    expected_score=0
    real_score=real_tracking[-1]
    assert real_param==expected_param
    assert real_score==expected_score

def test_find_move_direction():
   keys=['x','y']
   params = {'x': range(-10, 10), 'y': range(-10, 10)}
   upper_point={'x': 3, 'y': 16}
   lower_point = {'x': 2, 'y': 15}
   move_up={'x':True,'y':True}
   real_out_put=search_param._find_move_direction(fun=f,keys=keys,
                                                  params=params,
                                                  upper_point=upper_point,
                                                  lower_point=lower_point,
                                                  move_up=move_up)
   expected_out_put=(74.0,{'x':True,'y':False})

   assert real_out_put==expected_out_put

def test_init_upper_lower_points():
    params = {'x': range(-10, 10), 'y': range(-10, 10)}
    keys=list(params.keys())
    num_points={'x':20,'y':20}
    real_out_put_1,real_out_put_2=search_param._init_upper_lower_points(keys=keys,num_points=num_points)

    key_value_pairs=list(real_out_put_1.items())+list(real_out_put_2.items())
    assert all([key in keys and value>=0 and value<num_points[key] for key,value in key_value_pairs]) and \
            all([value2-value1==1 for value1,value2 in zip(list(real_out_put_1.values()),list(real_out_put_2.values()))])

def test_reset_upper_lower_points():

    params = {'x': range(-10, 10), 'y': range(-10, 10)}
    keys=list(params.keys())
    num_points=[20,20]
