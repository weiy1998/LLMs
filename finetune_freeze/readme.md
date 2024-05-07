## 预训练大模型（传统大模型）直接进行微调

> 该项目采用的基座大模型是 chatGLM3-6B，想要运行本项目，**请预先下载原生的chatGLM3-6B模型**

#### 直接微调参数的方法

- 微调所有层
- 调整部分参数（冻结部分层）
- 在模型尾部加一些层，只微调新加的层

#### 直接微调使用场景

1. 数据量大，tokens数量 >= 可调参数
2. 算立足，显存决定能否训练，显卡 flops 决定训练时间
3. 场景非常垂直

#### 为什么很少直接微调？（大模型调参难点）

1. 参数多，显存不足
2. 参数多，需要对应更大的训练数据
3. 参数多，不容易收敛
4. 参数多，调参时间过长，增加工期

#### 本项目直接微调流程

1. 将需要微调的数据处理成**问答对**的形式，以文旅数据为例：

    ~~~python
    {"context": "Instruction: 《梦里老家》实景演出地理位置\nAnswer: ", "target": "上饶市婺源县紫阳镇梦里老家文旅小镇"}
    {"context": "Instruction: 《梦里老家》实景演出介绍\nAnswer: ", "target": "《梦里老家》实景演出位于上饶市婺源县紫阳镇梦里老家文旅小镇。"}
    {"context": "Instruction: 《梦里老家》实景演出开放时间\nAnswer: ", "target": "3月4日-12月31日20:00-21:10(停止售票19:45,最晚入园19:55)"}
    {"context": "Instruction: 《梦里老家》实景演出门票价格\nAnswer: ", "target": "《梦里老家》实景演出门票价格：238元"}
    {"context": "Instruction: 《梦里老家》实景演出怎么样\nAnswer: ", "target": "好看的表演，第一次看还是值得一看的，下次还会来看"}
    {"context": "Instruction: 《梦里老家》实景演出中文名\nAnswer: ", "target": "《梦里老家》实景演出"}
    ~~~

2. 使用 tokenize_dataset_rows.py 对问答对进行分词化，token化，使用原生的 chatglm3 分词工具 

    ~~~bash
    python tokenize_dataset_rows.py --jsonl wenlv_data --save_path wenlv_token --max_seq_length 300
    ~~~

    > --max_seq_length: 问答对输入和输出的最大长度和；太长会增加显存；当长度超出设置值后，会自动截断超出部分

3. 使用 finetune_freeze.py 进行模型微调

    ~~~bash
    python finetune_freeze.py 
    ~~~

    >   dataset_path="data/wenlv_token"  微调数据路径
    >
    > output_dir  微调后的模型保存路径

4. 使用 test_freeze.py 测试微调后的模型

    ~~~bash
    python test_freeze.py
    ~~~

    > tokenizer  采用原生的chatglm3的分词器
    >
    > model  加载微调训练好的模型



##### 仅代表个人学习和工程实践途中个人观点，不喜勿喷，如有问题，请留言讨论！！！

