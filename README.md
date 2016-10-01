# MLND Capstone project
The capstone project is aiming to perform multi-digit number recognition on street view imagery.

## Dependencies
- Python 2.7
- NumPy/SciPy
- Matplotlib
- OpenCV
- Tensorflow

## How to run
The project has realized a neural network model to localize and detect digits in any given arbitrary image. Before running, download and uncompress the svhn\_multi\_digit and svhn\_region models in the saved\_models directory. Then, populate a directory with images to be evaluated by the model. Here, some sample images are stored in `sample/google_street_view_images` directory. Then run the following command:

`python multi_digit_recognition.py sample/google_street_view_images`

For each input image in the directory, the predicted localized region and digit sequence is displayed.

To evaluate the performance of the multi-digit detection model alone, populate already localized images into a directory, here `sample/google_street_view_images_cropped`, and run the following command:

`python multi_digit_recognition_wo_localization.py sample/google_street_view_images_cropped`

In the course of realizing the final model, several individual models were trained and evaluated. To run training on, for example MNIST multi-digit model, run:

`python mnist_multi_digit_train.py <RUN_NAME>`

where `<RUN_NAME>` can help identify its training logs and keep its model checkpoints in a clean directory. Make sure corresponding datasets have been downloaded and extracted in the manner specified in `dataset` directory. The SVHN models additionally take `-v` argument that optionally enables evaluating of validation set during training.

To evaluate a model, make sure the corresponding model checkpoints are present in `saved_models` directory and then run for example:

`python mnist_multi_digit_eval.py`

Alternatively, replace the model checkpoints trained with `<RUN_NAME>` during training in the eval scripts.

## Samples
Some correct predictions by the final model:
![positive](/doc/report/combined_predict_pos.png?raw=true "Positive") 

Some incorrect predictions by the final model:
![negative](/doc/report/combined_predict_neg.png?raw=true "Negative") 