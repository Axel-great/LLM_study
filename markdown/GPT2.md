## GPT2

### 1.dataclass

​	dataclass用于对类的简写，会自动创建def ___init___()进行参数初始化，用于更明确的定义函数数值

### 2.*Block

​	*为拆包指令,用于每个Block作为实例进行参数的传递，并且每个Block的参数为config，定义好GPT2主干网络中N层结构

~~~python
model = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
print(model)

'''
model = nn.Sequential(
    Block(config),
    Block(config),
    Block(config),
    Block(config)
)'''
~~~

### 3.Embedding和Linear

~~~python
import torch
import torch.nn as nn

# 假设 vocab_size=50000, embedding_dim=768
vocab_size = 50000
embedding_dim = 768
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# 假设 batch_size=4, seq_len=6（每个样本6个token）
x = torch.randint(0, vocab_size, (4, 6))  # 随机生成 token ID
print("Input shape:", x.shape)  # (4, 6)

# 通过 nn.Embedding
embeddings = embedding_layer(x)
print("Output shape:", embeddings.shape)  # (4, 6, 768)

# 输入X.shape(batch_size,seq_len)	输出Shape为(batch_size,seq,embedding_dim)
# Embedding的作用是在表长度vocab_size=50000的词典中，找到每个seq中tokens对应的向量
# 然后再将每一个tokens增加其特征表示的维度，也就是embedding_dim
~~~

​	Embedding为index的查找过程，将对应的index映射到指定的hidden_dim的维度中，相当于查找的一个过程，而nn.Linear是对数据进行线性变换的过程。

​	Embedding的查找表格，**可以是初始化未训练过的，也可以是已经训练好的Word2Vec**的表格。seq需要经过tokenizer编码成数字，然后进行Embedding。而这个查找表格就是需要学习的参数权重，最后会对这个参数权重进行更新。

​	如果使用权重共享，则Embedding的权重与输出层的Linear权重相同。由于Embedding层是没有偏置的，所以Linear层的最终输出也是没有的。

​	为什么可以共享?	-----因为Embedding层权重与Linear层的权重矩阵为转置关系。
$$
W_{\text{out}} = W_{\text{emb}}^T
$$
​	在初始时矩阵(batch_size,seq,vocab_size)--->(batch_size,seq,hidden_dim)，而输出层映射的结构是从:(batch_size,seq,hidden_dim)--->(batch_size,seq,vocab_size)，是一个**反向映射的过程**，所以二者权重矩阵功能相似，共享可以减少内存开销。

### 4.Loss计算

​	在模型**训练时**，target值是不能=0的，因为需要更新参数矩阵。

​	logit  (batch , seq_len, vocab_size)**--->** (batch * seq_len, vocab_size)  前面两个维度相乘相当于把每一个tokens放到一个矩阵当中，即原来的(seq_len , vocab_size) 矩阵中，不断向下拼接了许许多多的seq_len，然后这一个矩阵会包含所有的tokens。

~~~c#
原始 logits 形状：
[
    [[p1, p2, ..., p_vocab],  # 第一个句子的第一个 token
     [p1, p2, ..., p_vocab],  # 第一个句子的第二个 token
     ...],
     
    [[p1, p2, ..., p_vocab],  # 第二个句子的第一个 token
     [p1, p2, ..., p_vocab],  # 第二个句子的第二个 token
     ...]
]
~~~

~~~c#
view(batch * seq_len, vocab_size)之后
[
    [p1, p2, ..., p_vocab],  # 第 1 个 token
    [p1, p2, ..., p_vocab],  # 第 2 个 token
    ...
]
~~~

​	targets (batch , seq_len)  **--->**  (batch * seq_len) 变成了真实对应

~~~~csharp
原始 targets 形状：
[
    [5, 10, 25, ...],  # 第一个句子的 token 真实索引
    [3, 15, 27, ...],  # 第二个句子的 token 真实索引
    ...
]
~~~~

~~~c#
[5, 10, 25, 3, 15, 27, ...]  # 一维向量，每个值对应一个 token 的真实索引
~~~

​	使用交叉熵函数进行计算。logic矩阵先进行softmax，计算出tokens对应词汇表里面**每一个样本的概率**，在target里面查**看该tokens对应的真实概率**，最后进行Log计算，得出最后的loss值。

### 5.ModuleList和Sequential

- Sequential不需要调用forward()，可以直接执行
- ModuleList需要在forward中进行for循环调用，仅仅只是一个列表

### 6.self.apply()

多用于正态分布初始化，初始化是为了防止反向传播的时候梯度消失或者梯度爆炸

`nn.Module.apply(fn)` 是 `PyTorch` 提供的一个方法：

- **会递归地遍历**模型及其所有子模块。
- **对每个子模块执行 `fn(module)`**，其中 `fn` 是一个自定义的函数（这里是 `_init_weights`）。
- 适用于初始化参数、修改 `requires_grad` 状态等。
