## bug大全

### 1.Hugging Face登录问题

无法加载datasets，于是去huggingface登录获取tokens，之后发现login无法登录，找出以下解决方法。

1. 更新urllib3==1.25.11
2. 修改proxy的文件

[Pycharm与HuggingFace连接出现连接出现TLS/SSL connection has been closed (EOF) 问题的解决_huggingface-cli login-CSDN博客](https://blog.csdn.net/qq_59700461/article/details/134124983)

[huggingface(_hub)下载load报错ConnectionError: Couldn‘t reach ‘fusing/fill50k‘ on the Hub (SSLError)解决指南！_dataset hub connectionerror-CSDN博客](https://blog.csdn.net/qq_36525741/article/details/134417772)