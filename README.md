# lane_detection

`test_lanenet.py` is the file to generate images, json for the lane points and a video file with output displayed.  
Follow instructions from here https://github.com/MaybeShewill-CV/lanenet-lane-detection to set up the environment, clone the repo and download the model. Replace test_lanenet.py.  
Command to run it should be:  
```
python tools/test_lanenet.py --vid_path ./data/tusimple_vids_jl/tusimple1.mp4 --weights_path ./model/tusimple_lanenet/tusimple_lanenet.ckpt --save_dir ./data/tusimple_test_image_results1
```

lane_assignment.ipynb notebook contains demo code to get lane id given the lane points and bounding box coordinates.
