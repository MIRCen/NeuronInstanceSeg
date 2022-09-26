# A General Deep Learning framework for Neuron Instance Segmentation based on Efficient UNet and Morphological Post-processing
Huaqian Wu, Nicolas Souedet, Caroline Jan, Cédric Clouchoux, Thierry Delzescaux

Reference code for the paper [A General Deep Learning framework for Neuron Instance Segmentation based on Efficient UNet and Morphological Post-processing.](http://128.84.21.203/abs/2202.08682) 
Huaqian Wu, Nicolas Souedet, Caroline Jan, Cédric Clouchoux, Thierry Delzescaux. If you use this code or our datasets, please cite our paper:
```
@article{wu2022general,
  title={A General Deep Learning framework for Neuron Instance Segmentation based on Efficient UNet and Morphological Post-processing},
  author={Wu, Huaqian and Souedet, Nicolas and Jan, Caroline and Clouchoux, C{\'e}dric and Delzescaux, Thierry},
  journal={arXiv preprint arXiv:2202.08682},
  year={2022}
}
```

## Code


### Prerequisite
* Tensorflow
* Keras
* numpy
* tqdm
* OpenCV
* scikit-image

#### Training

The training set contains the following elements:
* images_all
* masks_all
* folds.csv

Optional: folds.csv attributes each image in the training set a label among [0, 1, 2, 3] for cross validation.

To train a model on a dataset located at `./datasets` and save the model weight to `./model`, use the following command:

```python train_efficient_b5.py -data ./datasets -model ./model```

#### Prediction
To predict images at `./test_data` using the model weight at `./model`, and save the prediction to `./prediction` use the following command:

```python predict_efficient_b5.py -i ./test_data -model ./model -o ./prediction```

#### Post-processing
The prediction of the neural network is a probability map, it requires a post-processing step to obtain the final instance segmentation:

```python postprocessing.py -i ./prediction/prob.png -o ./prediction/final_segmentation.png -r 10```
