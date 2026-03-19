# PUMBA
Repository for PUMBA, PUrely synthetic Multimodal/species invariant Brain extrAction

The first [pre-print](https://arxiv.org/abs/2505.07159) of the paper is out on arxiv!

More description on the method coming soon. Feel free to reach out on issues or email me at jp109 at iu dot edu or pjsjongsung at gmail.

Requirements
```
tensorflow (or torch)
scikit-image
dipy
```

Both the tensorflow and torch version can be run through

```python pumba_testing.py(or pumba_testing_torch.py) input_file_path output_path transform_method```

`transform_method` can be either `tranform_img` or `resize`.

`resize` is more stable, but `transform_img` calculates using an isotropic image, which are often beneficial with human images.

add ```--skip-postprocess``` in the end if you want to see the output without post processing.
