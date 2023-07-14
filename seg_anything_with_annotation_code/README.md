The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

or clone the repository locally and install with

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.
```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

To get pre-trained weights use:

```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Also install 

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx ftfy tqdm regex
```

To install image segmentation gui and annotation code run
```
pip install -r requirements.txt
```
And run the code using 

```
python click_segm.py --img <image_from_first_camera_pose> --i2 <image_from_second_camera_pose> --e1 <extrinsics_from_first_camera_pose> --e2 <extrinsics_from_second_camera_pose> --inparams <instrinsic_params> --incoeff <intrinsic_coefficients> --inmodel <intrinsic_distortion model> --inparams2 <instrinsic_params_for_second_pose> --incoeff2 <intrinsic_coefficients_for_second_pose> --inmodel2 <intrinsic_distortion_model_for_second_pose> --dimg <depth_image> --darr <depth_array>
```
