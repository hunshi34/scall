import torch
from transformers import AutoConfig, Phi3ForSequenceClassification, \
                         Phi3Config, AutoTokenizer

# Phi3ForSequenceClassification,

from transformers import TrainingArguments

output_dir='.checkpoint/results',          # 输出目录
num_train_epochs=3,              # 训练的总轮数
per_device_train_batch_size=8,   # 每个设备的训练批次大小
per_device_eval_batch_size=8,    # 每个设备的评估批次大小
warmup_steps=500,                # 学习率预热步数
gradient_accumulation_steps=2,
evaluation_strategy="no" ,
save_strategy= "steps" ,
save_steps= 50000 ,
save_total_limit= 1 ,
learning_rate= 2e-5 ,
weight_decay =0. ,
warmup_ratio =0.03 ,
lr_scheduler_type= "cosine" ,
logging_steps= 1 ,
tf32 =True ,
bits=None
model_max_length= 2048 ,
gradient_checkpointing= True ,
dataloader_num_workers= 4,
lazy_preprocess =True ,
report_to= "tensorboard" ,              # 日志记录步数
model_name = "/home/xh/llava_copy/checkpoints/microsoft_Phi-3-mini-4k-instruct/"


tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
bnb_model_from_pretrained_args = {}
if bits in [4, 8]:
    from transformers import BitsAndBytesConfig
    bnb_model_from_pretrained_args.update(dict(
        device_map={"": device},
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=bits == 4,
            load_in_8bit=bits == 8,
            llm_int8_skip_modules=["mm_projector"],
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_quant_type=quant_type # {'fp4', 'nf4'}
        )
    ))
model =Phi3ForSequenceClassification.from_pretrained(
    model_name ,
    config=config,
    attn_implementation="flash_attention_2",
    cache_dir=None,
    torch_dtype=torch.bfloat16
)
