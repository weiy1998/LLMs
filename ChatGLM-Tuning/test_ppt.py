
from transformers import AutoTokenizer, AutoModel
import json
#分词器 仍然用原生的
def extract_titile_prompts(paragraphs):
    results=[]
    for paragraph in paragraphs:
        if len(paragraph)==0:
            continue
        paragraph=[s.strip() for s in paragraph]
        paragraph=[s for i,s in enumerate(paragraph) if len(s)>10 or i==0]
        if len(paragraph)==0:
            continue
        if len(paragraph)==1:
            title=paragraph[0]
            prompts=[paragraph[0]]
        else:
            title=paragraph[0]
            prompts=paragraph[1:]
        results.append([title,prompts])
    return results
def convert_ppt(title_prompts,topic,describe):
    results={}
    directory=[title for title,_ in title_prompts]
    #第一页：目录
    results[0]={"titie":"目录","content":directory,"type":"directory"}
    #扩写具体内容
    page_num=1
    for title,prompts in title_prompts:
        results[page_num]={}
        results[page_num]["title"]=title
        results[page_num]["type"]="section"
        results[page_num]["content"]=[]
        for promot in prompts:
            text, _ = model.chat(tokenizer,"扩写{}".format(promot), history=[],do_sample=False)
            results[page_num]["content"].append({"sub_title":promot,"text":text})
        page_num+=1
    return results
def parser_response(response,topic,describe):
    paragraphs=response.split("\n\n")
    if len(paragraphs)==1:
        paragraphs=response.split("\n")
    paragraphs=[s.split("\n") for  s in paragraphs]
    title_prompts=extract_titile_prompts(paragraphs)
    result=convert_ppt(title_prompts,topic,describe)
    return result
import xlrd
from openpyxl import load_workbook
tokenizer = AutoTokenizer.from_pretrained("E:\code\chatglm\chatglm2", trust_remote_code=True)
#model = AutoModel.from_pretrained("E:\code\chatglm\chatglm2", trust_remote_code=True).cuda() 
# excel=load_workbook('产品介绍.xlsx')
# table = excel.get_sheet_by_name('Sheet1')
# schema=[]
# i=0
# results=[]
# for rx in table.rows:
#     if i==0:
#         schema=[rx[j].value.strip() for j in range(0,5)]
#         i+=1
#         continue
#     topic=rx[0].value.strip()
#     describe=" ".join([schema[j]+":"+rx[j].value.strip() for j in range(1,5)])
#     input="{}。根据上述信息，生成一个关于{}的大纲，生成二级标题".format(describe,topic)
#     response, _ = model.chat(tokenizer, input, history=[],do_sample=False)
#     result=parser_response(response,topic,describe)
#     results.append(json.dumps(result,ensure_ascii=False))
# with open("产品介绍ppt.jsonl","w",encoding="utf-8") as f:
#     f.writelines("\n".join(results))
result=tokenizer.encode("Instruction: 你是谁\nAnswer: ")
print (result)
result=tokenizer.encode("你是谁")
print (result)
# topic="智能文档"
# input="生成一个关于{}的大纲，生成二级标题".format(topic)
# response, _ = model.chat(tokenizer, input, history=[],do_sample=False)
# result=parser_response(response,topic)
# print (result)
