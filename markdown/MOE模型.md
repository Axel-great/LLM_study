## MOE 大全

### 1.MoE基础版

### 2.Sparse MoE 

### QA

#### 1.为什么Top_k可以被反向传播

​	反向传播的是传递Top_k参数的当前expert的概率，而非Top_k本身。其是离散值，不可微。通过softmax获得的概率值进行反向传播，可以得到Top_k的值。从而达到反向传播的效果

### 3.DeepSeek Moe

