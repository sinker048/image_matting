# Image_matting
## First 

Please put your data in demo/image_matting/input

## Next
  
Please download pretrained-model in prtrained


## Run
  

```
python -m demo.image_matting.inference --input-path demo/image_matting/input --output-path demo/image_matting/output --ckpt-path pretrained/modnet_photographic_portrait_matting.ckpt --portrait-path demo/image_matting/portrait
```

## Result

There will be mask in the `demo/image_matting/output`, and portrait in the `demo/image_matting/portrait`.
