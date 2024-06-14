import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer

class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_classes_task_a, num_classes_task_b):
        super(MultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier_task_a = nn.Linear(self.bert.config.hidden_size, num_classes_task_a)
        self.classifier_task_b = nn.Linear(self.bert.config.hidden_size, num_classes_task_b)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token output

        logits_task_a = self.classifier_task_a(cls_output)
        logits_task_b = self.classifier_task_b(cls_output)

        return logits_task_a, logits_task_b

def encode_sentences(sentences, model_name='paraphrase-MiniLM-L6-v2'):
    sentence_model = SentenceTransformer(model_name)
    embeddings = sentence_model.encode(sentences)
    return embeddings

def tokenize_sentences(sentences, tokenizer_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    return inputs['input_ids'], inputs['attention_mask']

def train_and_evaluate(model, input_ids, attention_mask, labels_task_a, labels_task_b, epochs=3):
    loss_fn_task_a = nn.CrossEntropyLoss()
    loss_fn_task_b = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits_task_a, logits_task_b = model(input_ids, attention_mask)

        loss_task_a = loss_fn_task_a(logits_task_a, torch.tensor(labels_task_a))
        loss_task_b = loss_fn_task_b(logits_task_b, torch.tensor(labels_task_b))

        loss = loss_task_a + loss_task_b
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        logits_task_a, logits_task_b = model(input_ids, attention_mask)
        preds_task_a = torch.argmax(logits_task_a, dim=1)
        preds_task_b = torch.argmax(logits_task_b, dim=1)

        print(f"Predictions for Task A: {preds_task_a}")
        print(f"Predictions for Task B: {preds_task_b}")

if __name__ == "__main__":
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A stitch in time saves nine.",
        "To be or not to be, that is the question."
    ]

    labels_task_a = [0, 1, 2]
    labels_task_b = [1, 1, 0]

    embeddings = encode_sentences(sentences)
    for sentence, embedding in zip(sentences, embeddings):
        print(f"Sentence: {sentence}")
        print(f"Embedding: {embedding}\n")

    input_ids, attention_mask = tokenize_sentences(sentences)

    model = MultiTaskModel('bert-base-uncased', num_classes_task_a=3, num_classes_task_b=2)
    train_and_evaluate(model, input_ids, attention_mask, labels_task_a, labels_task_b)
