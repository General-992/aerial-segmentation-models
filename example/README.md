
# Test Models on your Datasets

The `example\predict_labels` script take the model path and repo path (root) and outputs predicted images into the `root\outputs` if not specified.

### How to Use
```bash
 python -m example.predict_labels --repo $path_to_dataset$ --model-path $path_to_model.pth.tar$
```

### Dataset Structure Requirements

The dataset root directory must contain an `img` folder with the images and, optionally, a `msk` folder with the corresponding masks.

### Example Structure
```
/path_to_dataset/
├── img/ 
     │
     ├── image1.jpg │
     ├── image2.jpg │
     └── ... 
└── msk/ 
     │ (optional)
     ├── mask1.png 
     ├── mask2.png 
     └── ...
```
### Customizing the Dataset Class
The CustomDataset class is inherited from DatasetBase and is designed for flexibility. You can customize it as needed.
The \texttt{__init__} method in the CustomDataset class can be overridden to suit your specific repository or data arrangement. When making modifications, it is crucial to ensure that the self.files attribute retains the following structure:
```
self.files = [
    {'img': 'path_to_image1', 'msk': 'path_to_mask1'},
    {'img': 'path_to_image2', 'msk': 'path_to_mask2'},
    # ...additional image-mask pairs...
]
```
If the `msk` folder is absent, you may omit mask paths, but make sure that `self.files` consistently provides paths for the images and masks (when available) in this dictionary format.

### Command-Line Arguments
* --repo: Path to the dataset repository containing the img and (optionally) msk folders.
* --model-path: Path to the .pth file of the trained model.
* --gpu: GPU ID to use (default is 0).
* --output-label-path: Path to the output label directory. If not provided, it will default to output in the dataset root.




## Training on FLAIR


```bash
./torchconvs/datasets/download_flair1.sh

 python -m example.train -g 0 --model deeplab --batch-size 24 --data-path $PATH$

./view_log logs/XXX/log.csv
```


## Evaluating on FLAIR

```bash
 python -m example.evaluate -g 0 /example/logs/deeplab/model_best.pth.tar

./learning_curve example/logs/deeplab/log.csv
```



