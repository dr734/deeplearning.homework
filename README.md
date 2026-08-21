# 平面数据分类（单隐藏层）

原始笔记本（含全部代码）：
https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%203/Planar%20data%20classification%20with%20one%20hidden%20layer/Planar_data_classification_with_onehidden_layer_v6c.ipynb

## 更新说明（Updates to Assignment）

如果你之前在使用旧版本：
- 请点击右上角的 “Coursera” 图标以打开文件夹目录。
- 导航到文件夹：`Week 3/ Planar data classification with one hidden layer`。你可以在 `Planar_data_classification_with_onehidden_layer_v6b.ipynb` 中看到之前的工作。

修复与增强列表：
- 明确说明分类器将学习将区域分类为红色或蓝色。
- compute_cost 函数修复了 np.squeeze 的使用并将其结果转换为 float。
- compute_cost 的说明中澄清了 np.squeeze 的目的。
- compute_cost 中说明参数 `parameters` 并不需要，但保留在函数定义中以兼容自动评分器。
- nn_model 移除了参数值的显式提取，因为整个参数字典会传递给被调用的函数。

---

## 题目简介

这是第 3 周的编程作业：构建你的第一个具有一个隐藏层的神经网络。通过本作业你将学到：
- 实现一个带单隐藏层的二分类神经网络
- 使用像 tanh 这样的非线性激活单元
- 计算交叉熵损失
- 实现前向传播和反向传播

---

## 1 - 需要的包（Packages）

本 Notebook 使用的主要 Python 包：
- numpy（数值计算）
- sklearn（数据集与简单模型）
- matplotlib（绘图）
- testCases_v2（用于单元测试的例子）
- planar_utils（包含绘图、sigmoid、加载数据集等工具）

（代码导入在原笔记本中）

---

## 2 - 数据集（Dataset）

本练习使用一个“花朵”形状的二分类数据集，数据读取到变量 `X`（特征）和 `Y`（标签，红色：0，蓝色：1）。

可视化数据后，你会看到 `X` 的形状是 (2, 400)，`Y` 的形状是 (1, 400)，即共有 m = 400 个训练样本。

---

## 3 - 简单的逻辑回归（Simple Logistic Regression）

在尝试神经网络之前，先用 sklearn 的 LogisticRegressionCV 训练一个逻辑回归器并绘制其决策边界。结果显示逻辑回归在此数据集上的准确率约为 47%，说明数据并非线性可分，逻辑回归表现不佳。

---

## 4 - 神经网络模型（Neural Network model）

构建一个含单隐藏层的神经网络来解决该分类问题。模型结构图及数学表示如下（与原笔记本相同）：

对单个样本 x^(i)：
z^[1](i) = W^[1] x^(i) + b^[1]  
a^[1](i) = tanh(z^[1](i))  
z^[2](i) = W^[2] a^[1](i) + b^[2]  
â^(i) = a^[2](i) = σ(z^[2](i))  
y_prediction = 1 if a^[2](i) > 0.5 else 0

对所有样本的交叉熵损失：
J = -1/m Σ_{i=1}^m [ y^(i) log(a^[2](i)) + (1 - y^(i)) log(1 - a^[2](i)) ]

---

## 4.1 - 定义神经网络尺寸（Defining the neural network structure）

需要定义的三个变量：
- n_x：输入层大小（由 X 的形状确定）
- n_h：隐藏层大小（习题中将其设为 4）
- n_y：输出层大小（由 Y 的形状确定）

---

## 4.2 - 参数初始化（Initialize the model's parameters）

权重矩阵用小的随机数初始化（例如 np.random.randn(a,b) * 0.01），偏置向量初始化为 0。

返回参数字典：
- W1：形状 (n_h, n_x)
- b1：形状 (n_h, 1)
- W2：形状 (n_y, n_h)
- b2：形状 (n_y, 1)

---

## 4.3 - 前向传播（Forward propagation）

实现前向传播计算：
- Z1 = W1·X + b1
- A1 = tanh(Z1)
- Z2 = W2·A1 + b2
- A2 = sigmoid(Z2)

将 Z1, A1, Z2, A2 存入缓存（cache），供反向传播使用。

---

## 代价函数（Cost）

使用交叉熵损失计算代价 J（见上面的公式）。注意可能需要把结果从数组压缩为 float（使用 np.squeeze 并转为 float）。

---

## 反向传播（Backward propagation）

使用缓存中的 A1, A2 和参数 W2，按下列步骤计算梯度：
- dZ2 = A2 - Y
- dW2 = (1/m) dZ2 · A1^T
- db2 = (1/m) sum(dZ2)
- dZ1 = W2^T · dZ2 * (1 - A1^2)    （tanh 的导数）
- dW1 = (1/m) dZ1 · X^T
- db1 = (1/m) sum(dZ1)

返回 grads 字典：dW1, db1, dW2, db2。

---

## 参数更新（Parameter update）

使用梯度下降更新参数：
θ = θ - α * dθ

对 W1、b1、W2、b2 分别进行更新并将结果写回参数字典。

---

## 训练循环（The Loop / nn_model）

训练流程按以下顺序重复若干次（迭代）：
1. 前向传播计算 A2 和缓存
2. 计算代价
3. 反向传播计算梯度
4. 更新参数

训练完成后，使用学到的参数进行预测并可视化决策边界。

---

## 练习与测试（Exercises & Tests）

笔记本中为每个子函数（如 layer_sizes、initialize_parameters、forward_propagation、compute_cost、backward_propagation、update_parameters）提供了测试用例（testCases_v2）来验证实现的正确性。请在原始 Notebook 中运行这些单元以验证你的实现。

---

## 其他说明

- 本 README 翻译仅包括笔记本中的说明性内容与数学公式，代码单元保留在原始 Notebook 中并未翻译（请参阅上方链接以获得完整代码与可执行单元）。
- 如果需要，我可以把 Notebook 中的所有 Markdown 完整提取并转为独立的中文文档，或将翻译结果保存回仓库为 README.md（需你确认并提供目标仓库/分支权限）。
