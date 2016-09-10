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

def make_textbox(alg, param_lists):
    att_and_values = [i for i in alg.__dict__.iteritems() if i[0] in param_lists]
    storage = text_storage()
    for i, j in att_and_values:
        setattr(storage, i, FloatText(j))
    return storage

def algs_construction(argset, module_func, module_param=[]):
    algs_constructor = OC(argset, module_algs, module_func, module_param)
    algs = algs_constructor.construct()
    return algs


class TrackOptimizer(object):

    def __init__(self, input_path, module_func, module_param, FRAME=0):
        self.input_path = input_path
        self.module_func = module_func
        self.module_param = module_param
        self.FRAME = FRAME

    def run(self):
        self._prepare_prev_cells()
        self._prepare_curr_cells_till_last_alg()
        self._launch_interactive()

    def _prepare_prev_cells(self):
        self.instance, self.argset = self._prepare_instance(self.input_path, self.FRAME)
        self.algs = algs_construction(self.argset, self.module_func, self.module_param)
        self.instance.params.frame = self.FRAME
        self.instance.imgpath = self.instance.pathset[self.FRAME]
        self.instance._run_frame()

    @staticmethod
    def _prepare_instance(input_path, FRAME):
        setup = SettingUp(None, input_path)
        argset = setup.implement()
        instance = Caller(argset)
        instance._load_algs()
        instance.frame = FRAME
        instance.imgpath = instance.pathset[FRAME]
        instance._load_image()
        return instance, argset

    def _prepare_curr_cells_till_last_alg(self):
        # Second frame: run untill it reaches to the final tracking method
        self.instance.params.frame = self.FRAME + 1
        self.instance.imgpath = self.instance.pathset[self.FRAME + 1]
        self.instance._load_image()
        self.instance._prepare_curr_cells()
        for alg in self.algs[:-1]:
            alg.set_input(self.instance.image, self.instance.container, self.instance.params)
            alg.implement()
            self.instance.container = alg.return_output()


    def _launch_interactive(self):
        storage, eval_args = self._prepare_interactive_arguments()
        image, container, params = self.instance.image, self.instance.container, self.instance.params
        func = self.vis_child(image, container, params, self.argset, module_algs, (self.module_func[-1], ))  # used in eval
        w1 = eval(eval_args)
        display(w1)

    def _prepare_interactive_arguments(self):
        alg = self.algs[-1]
        param_lists = [i for i in dir(alg) if i.isupper()]
        storage = make_textbox(alg, param_lists)  # used in eval
        eval_args = create_eval_args(param_lists)
        return storage, eval_args

    @classmethod
    def vis_child(cls, image, container, params, argset, module_algs, module_func):
        def wrapper(**module_param):
            print 'loading...'
            c_cells = deepcopy(container.curr_cells)
            p_cells = deepcopy(container.prev_cells)
            mod_container = Container(argset, c_cells, p_cells)
            alg = algs_construction(argset, module_func, [module_param, ])[0]
            alg.set_input(image, mod_container, params)
            alg.implement()
            new_container = alg.return_output()
            cls.visualize_results(new_container)
        return wrapper

    @staticmethod
    def visualize_results(container):
        linked = len([i for i in container.curr_cells if i.previous])
        unlinked = len([i for i in container.curr_cells if not i.previous])
        print '{0} cells linked, {1} cells unlinked'.format(linked, unlinked)
