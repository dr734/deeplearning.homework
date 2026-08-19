# 用神经网络思维做逻辑回归

欢迎来到你的第一个（必做）编程作业！你将构建一个逻辑回归分类器来识别猫。这个作业会一步步引导你如何用神经网络的思路来完成它。

**说明：**
- 除非题目明确要求，否则在你的代码中不要使用循环（for/while）。

你将学到：
- 构建一个学习算法的一般架构，包括：
  - 初始化参数
  - 计算代价函数及其梯度
  - 使用优化算法（梯度下降）
- 将以上三部分按正确顺序整合到一个主模型函数中。

## <font color='darkblue'>更新</font>

这个 notebook 在过去几个月有过更新。先前版本名为 "v5"，当前版本现在命名为 '6a'

#### 如果你之前在旧版本上做过工作：
* 你可以在文件目录中查找旧文件（文件名中包含版本名）。
* 要查看文件目录，请点击这个 notebook 左上角的 "Coursera" 图标。
* 请将你在旧版本中的工作复制到新版本中，以提交评分。

#### 更新列表
* 前向传播公式，索引现在从 1 开始而不是 0。
* 优化函数注释现在写为 "print cost every 100 training iterations" 而不是 "examples"。
* 修正注释中的语法错误。
* 一致使用 Y_prediction_test 变量名。
* 绘图的轴标签现在写为 "iterations (hundred)" 而不是 "iterations"。
* 在测试模型时，测试图片通过除以 255 来归一化。

## 1 - 包

首先，运行下面的单元来导入此作业将用到的所有包。

- [numpy](www.numpy.org) 是 Python 的科学计算基础包。
- [h5py](http://www.h5py.org) 是用于操作以 H5 格式存储的数据集的常见包。
- [matplotlib](http://matplotlib.org) 是用来绘图的著名库。
- [PIL](http://www.pythonware.com/products/pil/) 和 [scipy](https://www.scipy.org/) 在最后用来测试你模型时加载你自己的图片。

```python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

%matplotlib inline
```

## 2 - 问题集概述

**问题描述**：你有一个数据集（"data.h5"），包含：
- 一个训练集，共 m_train 张图片，标注为猫（y=1）或非猫（y=0）
- 一个测试集，共 m_test 张图片，标注为猫或非猫
- 每张图片的形状为 (num_px, num_px, 3)，其中 3 是 RGB 三个通道。因此，每张图片是正方形（高度 = num_px，宽度 = num_px）。

你的任务是构建一个简单的图像识别算法，能够正确地将图片分类为猫或非猫。

让我们先熟悉一下数据集。运行下面的代码来加载数据。

```python
# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
```

我们在图像数据集（训练和测试）的变量名后添加了 "_orig"，因为我们将要对它们进行预处理。预处理后，我们会得到 train_set_x 和 test_set_x（标签仍然是 train_set_y/test_set_y）。

train_set_x_orig 和 test_set_x_orig 的每一行都是表示一张图片的数组。你可以运行下面的代码来可视化一个例子。也可以随意更改 index 的值来查看不同图片。

```python
# Example of a picture
index =10
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[0, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
```

许多深度学习中的软件错误来自矩阵/向量维度不匹配。如果你能把矩阵/向量的维度理清楚，就能避免很多常见错误。

**练习：** 找到下面的值：
- m_train（训练样本数量）
- m_test（测试样本数量）
- num_px（每张训练图片的高度/宽度）

注意 train_set_x_orig 的形状是 (m_train, num_px, num_px, 3)。例如，m_train 可以通过 train_set_x_orig.shape[0] 获得。

```python
### START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_x_orig.shape[0]
m_test =  test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
### END CODE HERE ###

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
```

**m_train, m_test 和 num_px 的预期输出**:
- m_train = 209
- m_test = 50
- num_px = 64

为了方便起见，你现在应该把形状为 (num_px, num_px, 3) 的图像重塑为形状为 (num_px * num_px * 3, 1) 的向量。之后，我们的训练（和测试）数据集就是一个包含这些向量的矩阵。

**练习：** 将训练集和测试集重塑，使每张 (num_px, num_px, 3) 的图片被展平为 (num_px * num_px * 3, 1)。

在把形状为 (a,b,c,d) 的矩阵 X 展平为形状为 (b*c*d, a) 的矩阵 X_flatten 时，一个技巧是使用：
```python
X_flatten = X.reshape(X.shape[0], -1).T      # X.T 是 X 的转置
```

```python
# Reshape the training and test examples

### START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3],train_set_x_orig.shape[0])
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[1]*test_set_x_orig.shape[2]*test_set_x_orig.shape[3],test_set_x_orig.shape[0])
### END CODE HERE ###

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
```

预期输出示例：
- train_set_x_flatten shape: (12288, 209)
- train_set_y shape: (1, 209)
- test_set_x_flatten shape: (12288, 50)
- test_set_y shape: (1, 50)
- sanity check after reshaping: 一组数值（用于检查）

为了表示彩色图像，每个像素有红、绿、蓝三个通道，每个通道的取值范围为 0 到 255。

一个常见的数据预处理步骤是中心化并标准化数据集，即减去全体 numpy 数组的均值，然后除以标准差或直接除以 255 将像素值缩放到 [0,1]。

```python
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
```

要记住的：
- 预处理新数据集的常见步骤包括：
  - 确认问题的维度形状（m_train, m_test, num_px 等）
  - 将数据集重塑，使每个样本变成大小为 (num_px * num_px * 3, 1) 的向量
  - 对数据进行归一化/标准化

## 3 - 学习算法的一般架构

现在设计一个简单算法来区分猫图像和非猫图像。

你将构建一个逻辑回归模型，用神经网络的思维方式来理解它。下面的图说明了为什么逻辑回归实际上是一个非常简单的神经网络！

<img src="images/LogReg_kiank.png" style="width:650px;height:400px;">

算法的数学表达式：

对于一个例子 x^(i)：
z^(i) = w^T x^(i) + b
ŷ^(i) = a^(i) = sigmoid(z^(i))
L(a^(i), y^(i)) = - y^(i) log(a^(i)) - (1 - y^(i)) log(1 - a^(i))

代价（cost）对所有训练样本求和后为：
J = (1/m) * sum_{i=1..m} L(a^(i), y^(i))

关键步骤：
- 初始化模型参数
- 通过最小化代价函数学习模型参数
- 使用学到的参数在测试集上做预测
- 分析结果并得出结论

## 4 - 构建算法的各个部分

构建神经网络的主要步骤：
1. 定义模型结构（如输入特征数）
2. 初始化模型参数
3. 循环训练：
   - 计算当前损失（前向传播）
   - 计算当前梯度（反向传播）
   - 更新参数（梯度下降）

通常你会把步骤 1-3 分别实现，然后把它们整合到一个名为 model() 的函数中。

### 4.1 - 辅助函数

练习：使用你在 "Python Basics" 中的代码，实现 sigmoid()。你需要计算 sigmoid(w^T x + b) = 1 / (1 + e^{-(w^T x + b)})

```python
# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """


    ### START CODE HERE ### (≈ 1 line of code)
    s = 1/(1+np.exp(-z))
    ### END CODE HERE ###
    
    return s
```

测试示例：
```python
print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
# 预期输出：sigmoid([0, 2]) = [0.5        0.88079708]
```

### 4.2 - 参数初始化

练习：实现参数初始化。你需要把 w 初始化为全零向量。如果不知道使用哪个 numpy 函数，请查阅 np.zeros()。

```python
# GRADED FUNCTION: initialize_with_zeros

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    

    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros((dim,1))
    b = 0
    ### END CODE HERE ###

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b
```

测试示例：
```python
dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))
# 预期输出：
# w = [[0.]
#  [0.]]
# b = 0
```

### 4.3 - 前向和反向传播

现在参数已初始化，你可以实施前向和反向传播来学习参数。

练习：实现 propagate() 函数，计算代价及其梯度。

提示：
- 前向传播：
  - 给定 X
  - 计算 A = sigmoid(w^T X + b)
  - 计算代价函数 J = -(1/m) * sum( y log a + (1-y) log(1-a) )

下面两个公式将被使用：
∂J/∂w = (1/m) X (A - Y)^T
∂J/∂b = (1/m) sum_i (a^{(i)} - y^{(i)})

```python
# GRADED FUNCTION: propagate
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A = sigmoid(np.dot(w.T,X) + b)              # compute activation
    cost = np.sum(((- np.log(A))*Y + (-np.log(1-A))*(1-Y)))/m  # compute cost
    ### END CODE HERE ###
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = (np.dot(X,(A-Y).T))/m
    db = (np.sum(A-Y))/m
    ### END CODE HERE ###

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
```

测试示例：
```python
w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
# 预期输出：
# dw = [[0.99845601]
#  [2.39507239]]
# db = 0.001455578136784208
# cost = 5.801545319394553
```

### 4.4 - 优化

- 你已经初始化了参数。
- 你也已经计算出代价函数和它的梯度。
- 现在，你需要用梯度下降来更新参数。

练习：实现优化函数。目标是通过最小化代价 J 来学习 w 和 b。对于参数 θ，更新规则为 θ = θ - α * dθ。

```python
# GRADED FUNCTION: optimize

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - (learning_rate*dw)
        b = b - (learning_rate*db)
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
```

测试示例（部分）：
```python
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
```

（可用 plt.plot(costs) 可视化代价随迭代变化）

练习：实现 predict() 函数，使用学到的 (w,b) 对数据集 X 进行预测。步骤：
1. A = sigmoid(w^T X + b)
2. 将 A 的每个元素转换为 0（如果激活 <= 0.5）或 1（如果激活 > 0.5），把预测结果存入 Y_prediction

```python
# GRADED FUNCTION: predict

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(w.T,X) + b)           # Dimentions = (1, m)
    ### END CODE HERE ###
    
    #### WORKING SOLUTION 1: USING IF ELSE #### 
    #for i in range(A.shape[1]):
        ## Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        #if (A[0,i] >= 0.5):
        #    Y_prediction[0, i] = 1
        #else:
        #    Y_prediction[0, i] = 0
        ### END CODE HERE ###
        
    #### WORKING SOLUTION 2: ONE LINE ####
    #for i in range(A.shape[1]):
        ## Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        #Y_prediction[0, i] = 1 if A[0,i] >=0.5 else 0
        ### END CODE HERE ###
    
    #### WORKING SOLUTION 3: VECTORISED IMPLEMENTATION ####
    Y_prediction = (A >= 0.5) * 1.0
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction
```

测试示例：
```python
w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(w, b, X)))
# 预期输出：predictions = [[1. 1. 0.]]
```

要记住的内容：
- 你已经实现了多个函数：
  - 初始化 (w,b)
  - 通过迭代优化来学习 (w,b)，包括计算代价及其梯度，并使用梯度下降更新参数
  - 使用学到的 (w,b) 对给定的样本集做预测

## 5 - 将所有函数合并为一个模型

现在把之前实现的各个模块按照正确顺序整合成 model() 函数。

练习：实现 model() 函数。使用以下命名：
- Y_prediction_test：测试集上的预测
- Y_prediction_train：训练集上的预测
- w, costs, grads：来自 optimize() 的输出

```python
# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    
    ### START CODE HERE ###
    
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
```

（注：为了可读性，这里保留了 notebook 中的所有代码块原样不变，中文 README 将 Markdown / 说明文本全部翻译为中文，代码保持与原 notebook 完全一致。）
