



from sklearn.model_selection import ParameterGrid as grid
import random
import numpy as np


def fl_search(fun:function,params:dict,n_iter:int)->dict:
    """
    This function is to actively search best parameters for one model, should be better than grid search.
    parameter space should be discrete and monotonic.
    :param params: The parameters space to search, with keys as parameters and values are list of candidate values
    :return: A tuple with first one is the dictionary for the best parameters, and second one is a list as the best score tracking
    """

    params = {'max_depth': [4, 5, 6, 7, 8, 9],
              'learning_rate': [0.03, 0.05, 0.1, 0.13, 0.15],
              'reg_lambda': [0.8, 1, 1.3, 1.5],
              'n_estimators': [80, 100, 150, 200]
              }

    keys=list(params.keys())

    num_points={}
    for key in keys:
        num_points[key]=len(params[key])
        params[key].sort()

    lower_point, upper_point=_init_upper_lower_points(keys=keys,num_points=num_points)
    move_up={}
    tracking=[]

    i=0
    while i<n_iter:
        # find the move direction for next round
        score,move_up= _find_move_direction(fun=fun,keys=keys,params=params,upper_point=upper_point,
                                          lower_point=lower_point,move_up=move_up)

        # Track the score for the optimization
        tracking.append(score)

        # Reset the lower_point and upper_point based move direction
        lower_point, upper_point = _reset_upper_lower_points(keys=keys, move_up=move_up,num_points=num_points,
                                                             upper_point=upper_point,lower_point=lower_point)
        i+=1

    param = {}
    for key in keys:
        if move[key]:
            param[key] = params[key][upper_point[key]]
        else:
            param[key] = params[key][lower_point[key]]

    return (param,tracking)

def _find_move_direction(fun:function,keys:list,params:dict,upper_point:dict,lower_point:dict,move_up:dict)->tuple:
    """
    This function is to calculate the best combination of upper_point and lower_point. The best one decide the moving
    direction for each parameter. The best score should be also stored.
    :param fun: One callable function to optimize, with inputs as the keys.
    :param keys: A list of strings, from the keys of params
    :param params: A dictionary containing all parameter space to search
    :param upper_point: A dictionary with keyse from params, values containing upper point index for each parameter
    :param lower_point: A dictionary with keyse from params, values containing lower point index for each parameter
    :param move_up: A dictionary with keyse from params, values are logic ones describing move up or down.
    :return: A tuple, with the first one is best score, and the second one is the moving direction move_up
    """
    best_score=np.Inf
    move_space={}
    for key in keys:
        move_space[key]=[False,True]

    for move in grid(move_space):
        param={}
        for key in keys:
            if move[key]:
                param[key]=params[key][upper_point[key]]
            else:
                param[key] = params[key][lower_point[key]]
        score=fun(**param)
        if score<best_score:
            move_up=move
            best_score=score
    return (best_score,move_up)

def _init_upper_lower_points(keys:list,num_points:dict)->tuple:
    """
    This function is to randomly initialize the upper_point and lower_point
    :param keys: A list of string from the keys of params
    :param num_points: A dictionary to store the number of point to search for each parameter
    :return: A tuple of two dictionary including upper_point and lower_point
    """
    lower_point,upper_point={},{}
    for key in keys:
        random_n=random.randint(0,num_points[key]-1)
        if random_n < num_points[key]-1:
            lower_point[key]=random_n
            upper_point[key]=random_n+1
        else:
            lower_point[key]=random_n-1
            upper_point[key]=random_n
    return (lower_point, upper_point)

def _reset_upper_lower_points(keys:list,move_up:dict,num_points:dict,upper_point:dict,lower_point:dict)->tuple:
    """
    This function is to reset the upper and lower sets for each parameters based on moving direction.
    :param keys: A list of string from the keys of params
    :param num_points: A dictionary to store the number of point to search for each parameter
    :param upper_point: A dictionary to store the upper point for each feature
    :param lower_point: A dictionary to store the lower point for each feature
    :return: A tuple of two dictionary including upper_point and lower_point
    """
    for key in keys:
        if move_up[key]:
            if upper_point[key]<num_points[key]-1:
                upper_point[key]+=1
                lower_point[key]+=1
        else:
            if lower_point[key]>0:
                upper_point[key]-=1
                lower_point[key]-=1
    return (lower_point,upper_point)