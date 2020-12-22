# image_matting
<h2>First 

Please put your data in demo/image_matting/input

<h2>Next
  
Please download pretrained-model in prtrained


<h2>Run
  

```
python -m demo.image_matting.inference --input-path demo/image_matting/input --output-path demo/image_matting/output --ckpt-path pretrained/modnet_photographic_portrait_matting.ckpt --portrait-path demo/image_matting/portrait
```

<h2>Result

There will be mask in the `demo/image_matting/output`, and portrait in the `demo/image_matting/portrait`.
