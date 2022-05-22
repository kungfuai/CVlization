# Data module


`DictDataset` (currently named `DataRows`) is a map-style dataset with `__getitem__` and `__len__` methods. `__getitem__` returns a python `dict` or a `dict`-like object.

`DictDataset` shares a familar interface with the pandas DataFrame. Using DataFrame to train a model is almost effortless:

```python

train_df, val_df = train_test_split(df, test_size=0.2) # dataset creation and splitting
model = LinearRegression() # model creation
model.fit(train_df[["feature1", "feature2"]]) # model training
val_df["prediction"] = model.predict(val_df[["feature1", "feature2"]]) # model prediction
print(accuracy_score(val_df["label"], val_df["prediction"])) # model evaluation
```

The same cannot be said about computer vision projects. Is it possible to have a rich DataFrame that logically holds image data, and expose a similar, friendly interface? This is what we are striving for in this data module.

A desirable interface looks like this:

```python

train_ds, val_ds = create_datasets() # images are not loaded into memory
train_ds[0]["img"] # fetch the image on the fly
train_ds["img"].ctype # rich and strong column type
train_ds["aug_img"], train_ds["aug_bboxes"]] = augment(train_ds["img"], train_ds["bboxes"]) # image and target augmentation
if sklearn_style:
    model = create_model(model_config) # e.g. a keras.Model, nn.Module, LightningModule
    model.fit(x=train_ds["aug_img"], y=train_ds["aug_bboxes"], val_x=val_ds["img"], val_y=val_ds["bboxes"]) # model training
else:
    # Different from mostly sklearn models, neural networks are often multi-task models.
    model_signature = dict(
        inputs=[
            {"key": "aug_img", "ctype": train_ds["aug_img"].ctype}
        ],
        targets=[
            {"key": "aug_bboxes", "ctype": train_ds["aug_bboxes"].ctype}
        ]
    )
    model = create_model(model_config, model_signature) # e.g. a keras.Model, nn.Module, LightningModule
    val_ds["aug_img"] = val_ds["img"]
    val_ds["aug_bboxes"] = val_ds["bboxes"]
    trainer.train(model, model_signature, train_ds, val_ds) # data loader is created inside this method
val_ds["pred_bboxes"] = model.predict(val_ds["img"]) # model prediction
mAP = Evaluator("mean_average_precision")
print(mAP(val_ds["bboxes"], val_ds["pred_bboxes"])) # model evaluation
```

- Computer vision datasets tend to be large and cannot be loaded in memory at once. So lazy loading and computation is a must.
- `ds[colum_name]` is itself a map-style dataset. `ds['img'][5]` will return the 6th image.
- Column assignment `ds['new_col'] = x` means `ds.assign('new_col', x)`. `ds[idx]['new_col']` and `ds['new_col'][idx]` should return `x[idx]`.
- `ds['pred'] = model(ds['img'])` is a shortcut for `ds.assign('pred', model(ds['img']))`.
- `model(ds['img'])` means `lazy_apply(model, ds['img'])` and returns a map-style dataset. No actual compute happens.
- Shouldn't a map-style dataset be named a LazyList or VirtualList?
