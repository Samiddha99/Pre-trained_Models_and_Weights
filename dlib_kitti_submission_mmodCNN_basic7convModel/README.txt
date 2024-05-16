This folder contains the contents of the dlib MMOD+CNN kitti car detection
submission.  The submission essentially shows what happens when you run dlib's
MMOD+CNN vehicle detection example program (i.e.
http://dlib.net/dnn_mmod_train_find_cars_ex.cpp.html) on the kitti 2D Car
detection benchmark (see http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).  

The trained detector is stored in the .dat files and the kitti_dets folder
contains the raw detection outputs which were submitted to the kitti evaluation
server.  The kitti_train_detector.cpp program is essentially a copy of
http://dlib.net/dnn_mmod_train_find_cars_ex.cpp.html except the detection
window has been made larger, the weight decay reduced, and early stopping (i.e.
use of test_one_step()) disabled since we just let it train on the whole kitti
training set until convergence rather than using early stopping based on a
validation split. 

The find_cars() function in kitti_train_detector.cpp was used to produce the
final detections.

This resulted in 83.14% average precision on the kitti moderate difficulty task.


If you want to understand how these tools work refer to the dlib documentation.
In particular, read this example program: http://dlib.net/dnn_mmod_train_find_cars_ex.cpp.html.  
The exact kitti training code in this folder is provided only for historical
reference so that the results of the experiment can be exactly reproduced if
desired, not as a source of general documentation.  See the dlib documentation
if you want to understand how it works (i.e. don't email me (Davis King) about
this without first consulting the documentation!).
