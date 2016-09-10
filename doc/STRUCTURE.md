## Structures

*covertrack/covertrack.py* goes through the following processes.  
```
setup = SettingUp(self.imgdir, self.ia_path)
outputdir = setup.implement()
PreprocessCaller(outputdir).run()
SegmentationCaller(outputdir).run()
TrackingCaller(outputdir).run()
compress_channel_image(outputdir)
PostprocessCaller(outputdir).run()
SubDetection(outputdir).run()
ApplyObjects(outputdir).run()
```

Given the input argument file, each of these callers outputs different files.  

__SettingUp__: It receives an input argument file and saves setting.json in the output folder.  

__Preprocessing__: It reads images from image folder and save images in *processed* folder.

__Segmentation__: It reads images from image folder or *processed* folder and produces the labeled images in *segmented* folder. These labeled images are saved as png 16-bit images, where each objects have a distinct values.  

__Tracking__: It reads images from *segmented* folder, and it produces the labeled images in *tracked* folder. Values of images correspond to ID of cells. Check how tracking goes with ImageJ (File->Import Sequences, Image->Lookup Tables->inverted glasbey)  

__Postprocess__: It reads raw images or processed images and labeled images from *tracked* folder, and it produces the labeled images in *objects* folder.  
Postprocess accounts for cleaning and stuff; removing short traces, gap closing and finding division events.  

__Subdetection__: It reads labeled images from *objects* folder and output more labeled images in *objects* folder. Subdetection for segment something not primary objects (cytoplasm, in many cases).

__ApplyObjects__: It reads labeled images from *objects* folder and raw images or processed images. It saves df.csv. This is to extract parameters from images.

As long as they follow the file format, you can substitute modules to some other software and skip those processes. For example, you can run DeepCell to save segmented images in *segmeneted* folder, and then run all the steps from Tracking. You can also manually segment or fix tracking...    
