
## HOW TO WRITE THE INPUT ARGUMENTS

Please take a look at *input_file* folder for examples.  

### Basics

Covertrack tyically goes through the following 7 modules.

1. Settingup
2. Preprocess
3. Segmentation
4. Tracking
5. Postprocess
6. Subdetection
7. ApplyObjects

From 1 to 6, you can specify what algorithms to use.  

In __*_operations.py__ (e.g. *covertrack/segmentation/segmentation_operations.py*), you will see a list of functions.  Each functions (or operations) defines a unique algorithm. So this file is a catalogue of functions.

In *covertrack/segmentation/segmentation_operations.py*, you will see a following example.
```
def example_thres(img, holder, THRES=100):
    return skilabel(img > THRES)
```

It means this function defines the foreground if pixel is above the constant THRES.  
In your input file, you can specify both  
(1) what function to use  
(2) what parameters to use

Take a look at *input_file/input_tests1.py* for example. These two are written in this manner.

```
segment_args = (dict(name='example_thres', THRES=500), )
```
If you are okay with the default arguments (THRES=100), then you don't need to specify.
```
segment_args = (dict(name='example_thres'), )
```
If you have a list of arguments, it will run them sequentially.  
For example, a following args in tracking will run 'run_lap', 'track_neck_cut' and 'watershed_distance' function.   
In tracking, it means they first try to associate objects in a previous frame to the current frame by running 'run_lap' algorithm, whatever not associated is then handled by 'track_neck_cut'... and so on.  
```
_param_runlap = dict(name='run_lap', DISPLACEMENT=50, MASSTHRES=0.2)
_param_tn = dict(name='track_neck_cut', DISPLACEMENT=50, MASSTHRES=0.2, EDGELEN=7)
_paramwd = dict(name='watershed_distance', DISPLACEMENT=50, MASSTHRES=0.2, ERODI=6)
track_args = (_param_runlap, _param_tn, _paramwd)
```



Similarly, if you want to add your function, you can add it by writing in any __*_operations.py__. Each modules have different requirements. Take a look at STRUCTURES.md.

### Tuning parameters
Normally parameter tuning requires the iterations of parameter setting and visually validation the results.  
Before running jobs in parallel (for this we use fireworks), I process just one folder untill it gives you good tracking results. You can tune parameters for each modules sequentially.  
e.g. If preprocessing looks nice, you can run `covertrack -3 input.py` and it will only run the segmentation. Check images created under _segment_ folder by eyes, tune parameters and run the same module again untill you get good result. 
Once segmentation looks good, you can move on to tracking and so on.  
See `covertrack --help` for other command line options.  

  
Optionally, _covertrack/notebooks_ folder provides some useful tools for parameter tunings for some modules.  
Open up `segment_optimization.ipynb` through jupyter notebook. This is the visualization tool where you can interactively try different operations and parameters. 

