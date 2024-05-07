from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
import os


tokenizer = AutoTokenizer.from_pretrained("/home/yw/code/chatglm3-6b", trust_remote_code=True)


@dataclass
class FinetuneArguments:
    dataset_path: str = field(default="data/alpaca")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


# 读取分词化的数据
def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        #问题+答案
        ids = feature["input_ids"]
        #问题长度
        seq_len = feature["seq_len"]
        # -100 特殊字符，表示不预测
        # [-100] * (seq_len - 1) “问题部分” 是不需要预测的
        # ids[(seq_len - 1) :] 预测答案
        # [-100] * (longest - ids_l) 补0的位置不需要预测
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,  # 模型输入
        "labels": labels,        # 模型输出
    }


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))

def mul(l):
    #[a,b]
    r=1
    for s in l:
        r*=s
    return r
os.environ["WANDB_DISABLED"] = "true"

def main():
    writer = SummaryWriter()
    # finetune_args, training_args = HfArgumentParser(
    #     (FinetuneArguments, TrainingArguments)
    # ).parse_args_into_dataclasses()
    
    training_args = TrainingArguments(output_dir="chatglm-6b-freeze",
                                    per_device_train_batch_size=8,
                                    remove_unused_columns=False,
                                    num_train_epochs=1)
    dataset_path="data/wenlv_token"
    # init model
    #默认加载的参数数据类型是float16 和pytorch transformers的版本有关
    model = AutoModel.from_pretrained(
        "/home/yw/code/chatglm3-6b", load_in_8bit=False, trust_remote_code=True, device_map="auto"
    ).cuda()#.to(torch.float32)
    #half float16
    #torch.float32  float32
    #cuda() 让模型在gpu中运行

    # 所有层全部冻结
    # for name, param in model.named_parameters():
    #     param.requires_grad=False
    
    # 每一层的形状
    # for name, param in model.named_parameters():
    #     print (name,param.requires_grad,param.shape)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  
    model.is_parallelizable = False
    model.model_parallel = False
    #model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    # load dataset
    dataset = datasets.load_from_disk(dataset_path)
    print (dataset)
    print(f"\n{len(dataset)=}\n")
    #model加载好的大模型
    
    # 层数
    layer_num = len([p for p in model.parameters()])
    frezze_para = 0  # 冻结参数
    train_para = 0  # 可训练参数
    for i, p in enumerate(model.parameters()):
        # 冻结最后两层前所有参数
        if i < layer_num - 2:
            p.requires_grad = False  # 冻结参数
            frezze_para += mul(p.shape)
        else:  # 最后两层可训练
            train_para += mul(p.shape)
    print("冻结参数：", frezze_para)
    print("可训练参数：", train_para)
    
    # print(dataset)

    model= model.to(torch.float32)
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,  # 微调数据
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,  # 将数据变为可训练形式
    )
 
    trainer.train()
    writer.close()

    # for name, param in model.named_parameters():
    #     if "0"  in name:
    #         print (param)

    保存模型
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
