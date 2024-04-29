
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

from peft import PrefixEncoder, PrefixTuningConfig
#分词器 仍然用原生的
tokenizer = AutoTokenizer.from_pretrained("E:\code\chatglm\chatglm2", trust_remote_code=True)
#加载训练好的模型模型
model = AutoModel.from_pretrained("E:\code\chatglm\chatglm2", trust_remote_code=True).cuda()
 
model = PeftModel.from_pretrained(model, "weather-lora").half()
model=model.eval()
print (model)

input="Instruction: 你是谁\nAnswer: "
response, history = model.chat(tokenizer, input, history=[])
print(response)

input="你是谁"
response, history = model.chat(tokenizer, input, history=[])
print(response)

