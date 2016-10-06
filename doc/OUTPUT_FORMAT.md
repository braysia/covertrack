
## The final output from ApplyObjects

The final output from the ApplyObjects module is `df.npz`.  
This npz file can be opened with `numpy.load`.
It contains the following three items:

- labels  
1d array of lists containing properties.  
e.g.
```
  [['nuclei', 'YFP', 'mean_intensity'],
   ['cytoplasm', 'CFP', 'area'],
   ['cell_id']]
```

- time   
1d array representing time. By default, time is a consecutive integers ranged from 0 to the last frame.  
e.g. `time = range(data.shape[2])` 

- data  
3-dimensional numpy array.  
The first dimension represents different properties. The order of properties are ordered with labels. 
The second dimension represents each cell.  
The third dimension represents time points. `data.shape[2] == len(time)`
