from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel

prompts = ['Ray Charles, the',
            'Grant Hill is a professional',
            'The law in Ikaalinen declares the language'
            ]
ground_truth = ['piano',
                'basketball',
                'Finnish'
                ]
target_new = ['violin',
              'soccer',
              'Swedish'
              ]
subject = ['Ray Charles',
            'Grant Hill',
            'Ikaalinen'
            ]

hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gpt2-xl')
editor = BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    subject=subject,
    keep_original_weight=True
)

print(metrics)
print('*'*20)

tokenizer = GPT2Tokenizer.from_pretrained('./hugging_cache/gpt2-xl')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'
generation_prompts = [
    "Ray Charles, the",
    "The law in Ikaalinen declares the language"
]

model = GPT2LMHeadModel.from_pretrained('./hugging_cache/gpt2').to('cuda')
batch = tokenizer(generation_prompts, return_tensors='pt', padding=True, max_length=30)

pre_edit_outputs = model.generate(
    input_ids=batch['input_ids'].to('cuda'),
    attention_mask=batch['attention_mask'].to('cuda'),
    max_length=10
)

post_edit_outputs = edited_model.generate(
    input_ids=batch['input_ids'].to('cuda'),
    attention_mask=batch['attention_mask'].to('cuda'),
    max_length=10
)
