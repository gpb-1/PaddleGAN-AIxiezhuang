# README
AI卸妆
PaddleGAN实现"AI卸妆"(更新paddlepaddle2.2.0)
===
***
一、项目背景介绍
---
针对目前多种批图软件对图像的处理和化妆，图片失真，即将使用paddlegan模型还原真实图片。


***

二、数据介绍
---

### ***2.1准备数据集***</br>

**CycleGAN需要的数据集形式如下:**

![](https://ai-studio-static-online.cdn.bcebos.com/f8a06161166643849980826bba2e2a0fd8c1601aa50147a3a8f1b2700d58cb24)
**其中 ：**</br>
**trainA为化妆后的图片：**
![](https://ai-studio-static-online.cdn.bcebos.com/a0e5b6fd50234168898f63f2112332753e92d988bfe14fb6a06235e771febbd3)
**trainB为卸妆后的图片:**</br>
![](https://ai-studio-static-online.cdn.bcebos.com/690989ecfcd44552afe329af60671727a11b1b115cd6403d87252471f6e3ac32)
**testA和testB为验证集，同理为化妆后和卸妆后的图片</br>
trainA.txt和trainB.txt为训练集图片路径，testA,testB同理**</br>
### ***2.2解压数据集***
    
#解压数据集<br>
`
!unzip data/data46804/dataset.zip -d /home/aistudio/dataset/
`

### ***2.3下载paddle的GAN[gan下载链接](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/gan)***
### ***2.4安装imageio和scipy模块***
`
!pip install -q imageio
`<br>`
!pip install -q scipy==1.2.1
`
三、模型介绍
---
GAN简介</br>
   生成对抗网络(Generative Adversarial Network[1], 简称GAN) 是一种非监督学习的方式，通过让两个神经网络相互博弈的方法进行学习，该方法由lan Goodfellow等人在2014年提出。
生成对抗网络由一个生成网络和一个判别网络组成，生成网络从潜在的空间(latent space)中随机采样作为输入，其输出结果需要尽量模仿训练集中的真实样本。
判别网络的输入为真实样本或生成网络的输出，其目的是将生成网络的输出从真实样本中尽可能的分辨出来。而生成网络则尽可能的欺骗判别网络，两个网络相互对抗，不断调整参数。
生成对抗网络常用于生成以假乱真的图片。此外，该方法还被用于生成影片，三维物体模型等。
CycleGAN和pix2pix的区别：
pix2pix需要成对(Paired)的图片来训练，进行图像翻译，即输入为同一张图片的两种不同风格，如下图左边部分所示:
CycleGAN可以利用非成对(Unpaired)的图片进行图像翻译，即输入为两种不同风格的不同图片，自动进行风格转换。如下图右边部分所示.传统的GAN是单向生成，
而CycleGAN是互相生成，网络是个环形，所以命名为Cycle。并且CycleGAN一个非常实用的地方就是输入的两张图片可以是任意的两张图片，也就是unpaired。</br>
![](https://ai-studio-static-online.cdn.bcebos.com/d3a0cbc4e03740968e00adfff63412a26cb9f231e6b140098c78db6e73158b92)</br>
**CycleGAN**由两个生成网络和两个判别网络组成，生成网络A是输入A类风格的图片输出B类风格的图片，生成网络B是输入B类风格的图片输出A类风格的图片。生成网络中编码部分的网络结构都是采用convolution-norm-ReLU作为基础结构，解码部分的网络结构由transpose convolution-norm-ReLU组成，判别网络基本是由convolution-norm-leaky_ReLU作为基础结构。生成网络损失函数由LSGAN的损失函数，重构损失和自身损失组成，判别网络的损失函数由LSGAN的损失函数组成。</br>
**CycleGAN的结构**如下：</br>
![](https://ai-studio-static-online.cdn.bcebos.com/17524bbfd90a4aa78deb43eb11c9b0f9524017f438184b6283498bbdef71077c)</br>
**Cycle-Gan**总结构有四个网络，第一个网络为生成（转化）网络命名为G:X---->Y；第二个网络为生成（转化）网络命名为F:Y--->X；第三个网络为对抗网络命名为Dx，鉴别输入图像是不是X；第四个网络为对抗网络命名为Dy，鉴别输入图像是不是Y。<br>

四、模型训练
---

### ***将制作好的数据集路径输入，并配置好训练参数：训练轮数，batchsize, checkpoint保存路径等，运行train.py(记得及时删除训练过程中产生的图片以及checkpoint,防止内存爆炸)***

```
!python -u PaddleGAN/tools/main.py --config-file PaddleGAN/configs/cyclegan_cityscapes.yaml 
```


五、模型评估
---
### 预测
#### 用生成的checkpoint进行“卸妆”，检验一下模型效果<br>
- --init_model 输入要用来预测的checkpoint路径
- --input_style 为A 表示转换风格为A(化妆后)-->B(卸妆后)，此处也可以反向生成，若--input_style 为B ,则转换风格为B(化妆前)-->A(化妆后)，即AI "上妆"
- --test_list 待预测图片路径存放在左侧test_list.txt文件内
- --输出结果保存在infer_result/cyclegan/文件夹下<br>


### 运行infer.py即可开始“卸妆”

```
!python PaddleGAN/tools/main.py --config-file PaddleGAN/configs/cyclegan_cityscapes.yaml --evaluate-only --load output_dir/cyclegan_cityscapes-2022-02-27-23-31/epoch_95_weight.pdparams
```
### 预测结果
#### AI卸妆效果如下
![原图](https://ai-studio-static-online.cdn.bcebos.com/d0e5a4a7d87647658b801c15ab5029d84bf6e226ab5d46f7986c0cc391e865af)
![“卸妆”结果](https://ai-studio-static-online.cdn.bcebos.com/12c976a5c5ec4a1a939ec27345522a4e15bb9bb95be544b6a392cff2450038f5)

六、总结与升华
---
目前处于初学阶段，如遇到任何问题，请留言交流，期待志同道合者，共同进步。

七、个人总结
---
paddlegan模型对于初学者非常有利于学习和掌握。paddle提供了许多轻量化模型，后期将对训练更多模型。
