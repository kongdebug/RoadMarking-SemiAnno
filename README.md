# Roadmarking SemiAnno

## 简介
- 目前已有很多针对车端传感器如行车记录仪、车载摄像头采集到的影像数据进行道路标线的标注的开源数据集，例如[BDD100K](https://bair.berkeley.edu/blog/2018/05/30/bdd/)、[ApolloScape](https://apolloscape.auto/lane_segmentation.html)等数据集，这为从影像中提取道路标线的工作开展提供了有效数据支撑
- 对于从无人机航拍影像中提取道路标线虽然也有了相应的工作，例如[SkyScapes ICCV2019](https://arxiv.org/abs/2007.06102)等，但没有将对应的数据集开源。若要对无人机影像提取道路标线做相关研究则需要自己标注，这是非常耗时耗力的
- **Roadmarking SemiAnno**是基于[Labelme](https://github.com/wkentaro/labelme)软件与[SAM-HQ](https://github.com/SysCV/SAM-HQ)视觉大模型的无人机影像道路标线半自动标注仓库。通过使用Labelme软件在影像中的道路标线上标注点或线，添加对应类别，点作为prompt传入SAM-HQ得到对应掩膜，最后根据掩膜与点的类别生成Labelme可读取的`.json`文件，以供人工修改，效果如下所示：


<table>
  <tr>
    <td>Labelme标注点或线</td>
    <td>SAM-HQ分割结果</td>
    <td>生成Labelme可读取文件</td>
  </tr>
  <tr>
    <td><img src="https://ai-studio-static-online.cdn.bcebos.com/5c4a936f3bec4493a42edbc0edde28a8391b012c8e884246901f92dc1458901b" width="250"></td>
    <td><img src="https://ai-studio-static-online.cdn.bcebos.com/b4b930f8c2fc4a96af45516efb8e73f98844373ccc224f2b9550ab7f4316b5c2" width="250"></td>
    <td><img src="https://ai-studio-static-online.cdn.bcebos.com/692d402822a445e5b1cd235bb5d5027f59bae86b0b6c429ca29435a73b5559da" width="250"></td>
  </tr>
</table>

## 如何安装

- 本仓库代码构建在`windows`系统上，要求 `python>=3.8`, 并且 `pytorch>=1.7` 与 `torchvision>=0.8`。请按照[PyTorch官网](https://pytorch.org/get-started/locally/)的说明安装PyTorch和TorchVision依赖项。强烈建议同时安装PyTorch和TorchVision并支持CUDA
- 在本地克隆仓库，并使用以下语句安装`Roadmarking SemiAnno`:

```shell
git clone https://github.com/kongdebug/RoadMarking-SemiAnno.git
cd RoadMarking-SemiAnno
pip install -r requirements.txt
```

- 在`RoadMarking-SemiAnno`文件夹下新建`pretrained_checkpoint`文件夹，并根据自己需要从谷歌云盘下载对应的权重放入该文件夹下，模型类别与链接如下所示。同时在[AI Studio](https://aistudio.baidu.com/datasetdetail/250409)也上传对应权重可供下载
    - `vit_b`: [ViT-B HQ-SAM model.](https://drive.google.com/file/d/11yExZLOve38kRZPfRx_MRxfIAKmfMY47/view?usp=sharing)
    - `vit_l`: [ViT-L HQ-SAM model.](https://drive.google.com/file/d/1Uk17tDKX1YAKas5knI4y9ZJCo0lRVL0G/view?usp=sharing)
    - `vit_h`: [ViT-H HQ-SAM model.](https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?usp=sharing)
    - `vit_tiny` (**Light HQ-SAM** 是与Mobile-SAM对标的): [ViT-Tiny HQ-SAM model.](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth)


## 使用流程

### 1. 使用Labelme标注点与类别

- 双击`RoadMarking-SemiAnno`文件夹下的`Labelme.exe`，打开Labelme软件。然后以`Point`、`Line`、`Linestrip`进行对图像中的道路标线进行标注
- 推荐使用`Point`标注虚线类，`Line`标注简单的箭头类如直行、右转等，`Linestrip`标注复杂一些的标线

**注**：对于连续类型的标线，即实线类，本仓库的效果还有待提高

### 2. 使用SAM-HQ模型生成掩膜

- 通过执行以下命令将标注的点转换为语义分割数据集的Label掩膜

```shell
python semi-anno\pointsjson2masks.py --json_dir $json_dir$ --output_dir $output_dir$ --labels $labels$ --vis_dir $vis_dir$ --annotation_dir $annotation_dir$ --ext $ext$ --model-type $model_type$ --patch_size $patch_size$
```

- 参数释义如下：
    - `json_dir`: Labelme标注的点保存成json文件所在的文件夹路径
    - `output_dir`: 根据标注点自动生成语义分割掩膜json文件保存的文件夹，需要与要标注的图像在同一文件夹
    - `labels`: 标注类别文件的路径， 需要为`.txt`文件且第一行为'_background_'
    - `vis_dir`: 掩膜结果可视化保存路径
    - `annotation_dir`: 掩膜标签影像，单通道，可以直接用于[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)与[]()这两个主流语义分割模型仓库
    - `ext`: 待标注的图像的后缀
    - `model_type`: 根据点自动化生成掩膜的模型类型，与下载的权重类型对应
    - `patch_size`: 即以标注的关键点为中心，从完整图像中切割图像块传入SAM-HQ模型中的图像块大小

**注意**：`RoadMarking-SemiAnno\exp_data`为示例数据，`pointsjson2masks.py`脚本中默认参数与示例数据相一致，释义若还不清楚可对照默认参数与示例数据

### 3. 通过labelme检查或修改生成结果

- 完成自动化生成语义分割掩膜标注的json文件后，再次双击`Labelme.exe`并打开标注图像文件夹，就可以查看生成的结果，并对结果进行人工检查或直接修改

## 致谢

- 本项目参考以下仓库，感谢他们的开源：
    - Labelme: https://github.com/wkentaro/labelme
    - SAM-HQ: https://github.com/SysCV/SAM-HQ 

## 联系

- 如果在使用过程中遇到问题，欢迎提issue或者联系：KeyK@foxmail.com
