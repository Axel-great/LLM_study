### Git

下载代码 

git clone --recursive git@gitee.com:wang-66/diff-power.git --depth 10 下载前十次的提交 --depth 选择下载提交次数，避免文件过大

[

git submodule update --init --recursive  **对于一些很久的分支，要用这个更新一下子模块**

git status **查看状态,确定一下是否有没有track的子模块**

]

git config --global user.email "hhh"  **查看谁更改了此文件**

git config --global user.name "happy-yan"



git checkout -b happy/0817_test

git add xx.py 增加某一个py文件的修改

git commit -m "[planner]: add st boundary pre decider test" **添加评论，可以直接在vscode里面直接操作**

git push --set-upstream origin happy/0817_test  首次分支代码提交到仓库



git reset --soft HEAD^	代码编辑回退

git push -f	编辑代码回退后再次修改的强制提交



git remote -v

git remote add origin git@gitee.com:wang-66/altc.git

git fetch --depth 10



### Transformer初步简单实现

#### 1.在实现之前需要对每个数据进行Linear初始化

​	linear初始化是输入X，然后生成相同的dim进行Wq、Wv、Wk的变换矩阵的计算，生成之后的QKV。**==Linear的本质其实就是对矩阵进行线性变换+偏置系数（这里没有加偏置），矩阵相乘的本质也是线性变换==**

```python
self.query_proj = nn.Linear(hidden_dim,hidden_dim)
self.key_proj = nn.Linear(hidden_dim,hidden_dim)
self.value_proj = nn.Linear(hidden_dim,hidden_dim)
```

#### 2.Softmax需要指定维度

​	Softmax是对每一个行向量进行归一化计算，进行概率预测，所以需要对行进行归一化。而Q*V^T之后得到矩阵的格式为(batch_size,seq_len,seq_len)也就是对第二个维度归一化。在代码中为**dim=-1**，作用在 **列方向**（每一行进行 Softmax）。

~~~python
import torch
from torch import nn
import math
from torch import functional as F
class selfAttention(nn.Module):
    def __init__(self, hidden_dim:int = 728)->None:
        super().__init__()
        self.hidden_dim = hidden_dim
    #初始化qkv
        self.query_proj = nn.Linear(hidden_dim,hidden_dim)
        self.key_proj = nn.Linear(hidden_dim,hidden_dim)
        self.value_proj = nn.Linear(hidden_dim,hidden_dim)
    
    def forward(self,X):
        # Q : (batch_size,seq_len,dim)
        # K*T:(batch_size,dim,seq_len)
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)

        attention_value = torch.matmul(
            #给k转置
            Q,K.transpose(-1,-2)
        )

        attention_softmax = torch.softmax(
            attention_value / math.sqrt(self.hidden_dim),dim=-1
        )

        result = torch.matmul(
            attention_softmax,V
        )
        return result

X = torch.rand(2,3,4)
self_att_test = selfAttention(4)
self_att_test(X)

~~~

$$
\text{Attention}(Q, K, V) = \text{softmax} \left(\frac{Q K^T}{\sqrt{d_k}} \right) V
\
$$



### Transformer多头注意力机制

#### 1.多头注意力机制区别

​	多头注意力机制相比普通，多了concat的内容，通过多个不同的Wq、Wk、Wv矩阵构造出多个不同的QKV，之后进行组合做最后的线性变换。

#### 2.多头注意力机制流程

- **初始化 QKV 的 Linear 变换**
- **对 QKV 进行拆分**  
   - 原形状：`(batch, seq, hidden_dim)`  
   - 拆分后：`(batch, seq, head_num * head_dim)`  
- **调整维度顺序**  
   - 变换为：`(batch, head_num, seq, head_dim)`  
- **对 K 进行转置**  
   - 计算 `Q * K^T / d`  
- **对数据进行 Mask 处理**
- **计算 Softmax 并进行 Dropout**
- **输出最终注意力结果**

### QA

##### 1.为什么要进行开根号

​	不开根号softmax会将很大的值拉到无穷，导致其余数几乎为零，导致**梯度消失**

##### 2.dropout位置

​	dropout 位置在Softmax之后，而不是在乘完V之后。

##### 3.mask位置和大小

​	在softmax之前加入mask，屏蔽掉填充的0数值，给一个无限小的值。masked_fill（）填充。如果只**是扩展视图用expand（）节省内存**，repeat（）是复制数据！用于计算优先expand（）

​	mask大小要和Q*K之后的维度一致，即batch_size,seq，seq

##### 4.view和reshape

​	view更符合pytorch的写法，**不改变数据存储，但是需要内存连续**。所以在transpose之后需要加一个contiguous()函数

##### 5.mask和dropout层

​	mask需要在softmax之前，dropout需要在softmax之后。mask用于屏蔽padding的值，防止对softmax计算有影响

##### 6.🎯 LayerNorm和Softmax的区别

✅ **LayerNorm** 让特征稳定，**防止梯度问题**，但**不会改变输入的数值关系**。  
✅ **Softmax** 计算 **概率分布**，让所有值加起来等于 `1`，用于分类和注意力机制。  

🚀 **LayerNorm 主要用于网络内部，Softmax 主要用于输出概率！🔥**
