# NeuralStyleTransfer



Content Image                           |  Style Image                   | Result            
:--------------------------------------:|:------------------------------:|:-------------------------:
<img src=content_images/content_image2.jpg width="200" height="250"/> |  <img src=style_images/style2.jpg width="200" height="250"/> |  <img src=Result/target2.jpeg width="200" height="250"/>
<img src=content_images/content_image5.jpg width="200" height="250"/> |  <img src=style_images/style5.jpg width="200" height="250"/> |  <img src=Result/target4.jpg width="200" height="250"/>
<img src=content_images/content_image3.jpg width="200" height="250"/> |  <img src=style_images/style3.jpg width="200" height="250"/> |  <img src=Result/target3.jpeg width="200" height="250"/>


An implementation of neural style in PyTorch.


The class takes in the file path of the content image and the style image form the respective folders and generates the result in the result folder. Few examples have been provided.

I have added support for GPU. Estimated time for 10000 iteration when run on the Google colab GPU is 700 seconds. Feel free to try ne styles and images. I have only run on portraits and have recieved good results.

