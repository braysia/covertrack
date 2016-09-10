from settingup.settingup import SettingUpCovert as SettingUp
from module import OperationsConstructor as OC
from tracking import tracking_algs as module_algs
from ipywidgets import *
from traitlets import link
from IPython.display import display
import matplotlib.pyplot as plt
# from traitlets import link
from tracking.call_tracking import TrackingCaller
from copy import deepcopy
from tracking.track_utils.cell_container import Container

Caller = TrackingCaller

class text_storage(object):
    pass

def create_eval_args(param_lists):
    '''
    e.g.
    >>> param_lists = ['PAR1', 'PAR2]
    'widgets.interactive(func,PAR1=storage.PAR1,PAR2=storage.PAR2)'
    '''
    list_param_lists = [''.join([i, '=storage.', i]) for i in param_lists]
    eval_args = ','.join(list_param_lists)
    eval_args = 'widgets.interactive(func,' + eval_args + ')'
    return eval_args


def algs_construction(argset, module_func, module_param=[]):
    '''return the first alg. will be modified in the future'''
    algs_constructor = OC(argset, module_algs, module_func, module_param)
    algs = algs_constructor.construct()
    return algs


def make_textbox(alg, param_lists):
    att_and_values = [i for i in alg.__dict__.iteritems() if i[0] in param_lists]
    storage = text_storage()
    for i, j in att_and_values:
        setattr(storage, i, FloatText(j))
    return storage


def interactive_track(input_path, module_func, module_param, FRAME=0):
    # First frame
    instance, argset = prepare_instance(input_path, FRAME)
    algs = algs_construction(argset, module_func, module_param)
    instance.params.frame = FRAME
    instance.imgpath = instance.pathset[FRAME]
    instance._run_frame()

    # Second frame: run untill it reaches to the final tracking method
    instance.params.frame = FRAME + 1
    instance.imgpath = instance.pathset[FRAME + 1]
    instance._load_image()
    instance._prepare_curr_cells()

    container = instance.container

    for alg in algs[:-1]:
        alg.set_input(instance.image, container, instance.params)
        alg.implement()
        container = alg.return_output()
        print len([i for i in container.curr_cells if i.previous is not None])

    alg = algs[-1]
    single_module_func = (module_func[-1], )
    param_lists = [i for i in dir(alg) if i.isupper()]
    storage = make_textbox(alg, param_lists)  # used in eval
    eval_args = create_eval_args(param_lists)
    
    func = vis_child(instance.image, container, instance.params, argset, module_algs, single_module_func)  # used in eval
    w1 = eval(eval_args)
    display(w1)

    
def vis_child(image, container, params, argset, module_algs, module_func):
    def wrapper(**module_param):
        c_cells = deepcopy(container.curr_cells)
        p_cells = deepcopy(container.prev_cells)
        mod_container = Container(argset, c_cells, p_cells)
        alg = algs_construction(argset, module_func, [module_param, ])[0]
        alg.set_input(image, mod_container, params)
        alg.implement()
        new_container = alg.return_output()
        visualize_results(new_container)
    return wrapper


def prepare_instance(input_path, FRAME):
    setup = SettingUp(None, input_path)
    argset = setup.implement()
    instance = Caller(argset)
    instance._load_algs()
    instance.frame = FRAME
    instance.imgpath = instance.pathset[FRAME]
    instance._load_image()
    return instance, argset


def call_vis(input_path, segment_func):
    instance, argset = prepare_instance(input_path)
    vis(instance, segment_func, argset)


def visualize_results(new_container):
    linked = len([i for i in new_container.curr_cells if i.previous])
    unlinked = len([i for i in new_container.curr_cells if not i.previous])
    print '{0} cells linked, {1} cells unlinked'.format(linked, unlinked)

