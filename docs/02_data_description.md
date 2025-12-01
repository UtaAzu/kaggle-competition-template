Dataset description template

Please document the dataset layout for your competition/project. For example:

- `train.csv` - optional metadata table for training
- `train_images/` - folder containing training images
- `train_masks/` - folder containing ground-truth masks (if segmentation is required)
- `test_images/` - folder for test images ready for inference
- `sample_submission.csv` - template for submission structure

Notes:
- If you use Kaggle dataset uploads, set `KAGGLE_DATASET_SLUG` and adapt the scripts' CLI options / environment variables to point to the dataset.
-- This template intentionally does not prescribe a particular dataset layout; keep it minimal and add examples in `examples/` for reference.