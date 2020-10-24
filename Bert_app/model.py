import numpy as np
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch


model = BertForSequenceClassification.from_pretrained("bert_model_save") # kaydedilen model

tokenizer = BertTokenizer.from_pretrained('bert_token_save', do_lower_case=True) # kaydedilen tokenizer

max_len = 280

def tokenize(text): # raw text verisini tokenize etmek için fonksiyon

	# Notebook üzerinde tokenizasyon ile ilgili açıklamlar mevcut
	# Buradaki fonksiyonun tek farkı veri seti için for loop oluşturmak yerine
	# tek bir string üzerinde kullanmak üzere fonksiyon tanımlıyoruz

	encoded_text = tokenizer.encode_plus(
                        text,                     
                        add_special_tokens = True, 
                        max_length = max_len,          
                        pad_to_max_length = True,
                        return_attention_mask = True,  
                        return_tensors = 'pt',   
                   )

	input_ids = []
	attention_masks = []

	input_ids.append(encoded_text['input_ids'])
	attention_masks.append(encoded_text['attention_mask'])

	input_ids = torch.cat(input_ids, dim=1)
	attention_masks = torch.cat(attention_masks, dim=1)

	return input_ids, attention_masks

def predict(input_id, attention_mask): # tokenize edilmiş değerleri modelde kullanıp tahmin yürütmek için tanımlanan fonksiyon
	output = model(input_id, token_type_ids=None, 
                      attention_mask=attention_mask)

	logit = output[0]
	logit = logit.detach().cpu().numpy()

	prediction = []
	prediction = np.argmax(logit, axis=1).flatten()

	if prediction[0] == 0:
		return("Hesap")
	elif prediction[0] == 1:
		return("İade")
	elif prediction[0] == 2:
		return("İptal")
	elif prediction[0] == 3:
		return("Kredi")
	elif prediction[0] == 0:
		return("Kredi Kartı")
	else:
		return("Müşteri Hizmetleri")



