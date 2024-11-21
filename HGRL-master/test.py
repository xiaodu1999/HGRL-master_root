from transformers import T5ForConditionalGeneration, T5Tokenizer
import transformers

# 加载 T5 模型和分词器
model_name = "t5-small"  # 您可以替换为其他 T5 版本，如 "t5-small", "t5-large", "t5-3b" 等
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# 打印模型的配置
print("Model Config:", model.config)

# 打印 transformers 库的版本
print("Transformers Version:", transformers.__version__)
