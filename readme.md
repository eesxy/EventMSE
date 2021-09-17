# Event质量评价

## Update

- 各local patch的MSE值计算方式修改为stack不叠加，而是在各stack上分别计算后平均

- 输入方式改成按event数划分patch，单个patch的events数量按照图像尺寸\*density控制，默认density为0.1
- 仅训练了3D U-Net

## 数据预处理

数据预处理部分在matlab上进行

*注：预处理代码含有部分硬链接，需要修改后使用*

- `./preprocess/raw2mat.m `：借助`metavision_evt3_raw_file_decoder`模块将prophesee数据转成ev结构
- `./preprocess/mat2stack.m`：将上一步处理好的数据转成模型输入的stack形式
- `./preprocess/train_dataset.m`：将stack形式数据切分并保存为训练集
- `./preprocess/test_dataset.m`：不切分直接保存为测试集
- 其他辅助函数

  - `lp_save`：用来保存数据

## 模型训练

模型训练在python上实现

### 3D U-Net

- `trainer.py`：训练驱动

  - 损失函数为MSE
  - 初始学习率1e-2，scheduler使用ReduceLROnPlateau检测loss
  - 使用L2正则化防止过拟合，weight_declay=1e-2
  - 训练集：验证集比例为8：2
  - 在43epoch时早停

  *注：由于使用了benchmark加速，结果可能无法完全复现*

- `model.py`：模型，4次下采样，numFilters=16，图像尺寸事先扩大为16对齐

- `dataset.py`：dataloader使用的自定义dataset类

- `main.py`：主函数

- `./model`、`model.pth`：各epoch的模型和最终选定的模型

效果图：

## 使用

- `./usage/usage.py`：集成了UNet_3D, Transformer, Scorer三个类以及使用例

  - `Transformer`：将n*4矩阵转换成网络输入，也可以读入txt格式输入

    > 可以设置stack_dense参数和threshold参数
    >
    > stack_dense：若所给输入的有效尺寸明显小于总尺寸，比如拍摄小于窗口尺寸的照片，可能需要调节此参数使其适应有效尺寸
    >
    > threshold：prophesee相机容易出现较大的脉冲噪声，泛化模型暂时无法完全克服此类噪声，如有较大影响可使用此参数，一般设置为3~5即可完全消除脉冲噪声的影响

  - `Scorer`：用来读入模型并进行预测，给出打分或作图

