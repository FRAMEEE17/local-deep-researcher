import torch
import torch.nn as nn
import pickle
from transformers import AutoTokenizer, XLMRobertaModel
from typing import Dict, Any
from .configuration import Configuration
import time

class CustomXLMRobertaModel(nn.Module):    
    def __init__(self, num_labels):
        super(CustomXLMRobertaModel, self).__init__()
        model_name = 'symanto/xlm-roberta-base-snli-mnli-anli-xnli'
        self.roberta = XLMRobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_labels)
        )
        self.loss = nn.CrossEntropyLoss()
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        output = self.dropout(output.pooler_output)
        logits = self.classifier(output)

        if labels is not None:
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class IntentClassifier:    
    def __init__(self, config: Configuration):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        
        # Load label encoder (saved with torch.save)
        self.label_encoder = torch.load(config.label_encoder_path, map_location='cpu')
        
        # Load model (4 classes from your training)
        self.model = CustomXLMRobertaModel(num_labels=4)
        self.model.load_state_dict(torch.load(config.intent_model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Intent classifier loaded on {self.device}")
        print(f"Available intents: {list(self.label_encoder.classes_)}")
    
    def classify_intent(self, text: str) -> Dict[str, Any]:
        """Classify the intent of input text (following your notebook implementation)"""
        start_time = time.time()
        
        # Tokenize 
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            probabilities = torch.softmax(outputs['logits'], dim=-1)
            prediction = outputs['logits'].argmax(dim=-1).item()
            confidence = probabilities[0][prediction].item()
        
        # Decode using your trained label encoder
        intent_class = self.label_encoder.inverse_transform([prediction])[0]
        
        # Map to routing strategy based on your trained classes
        routing_strategy = self._map_intent_to_strategy(intent_class)
        
        processing_time = time.time() - start_time
        
        return {
            "intent": intent_class,
            "confidence": confidence,
            "routing_strategy": routing_strategy,
            "processing_time": processing_time,
            "prediction_label": prediction,  # 0,1,2,3
            "all_probabilities": {
                self.label_encoder.inverse_transform([i])[0]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }
        }
    
    def _map_intent_to_strategy(self, intent_class: str) -> str:
        """Map intent class to search strategy following the trained model"""
        # Based on your xlm_roberta_ft.ipynb label mapping:
        # 0: Academic Research Query
        # 1: Casual Conversation and General Query  
        # 2: Hybrid Research Query
        # 3: Web Search and Current Information Query
        
        intent_mapping = {
            "Academic Research Query": "arxiv_search",           # Label 0 → ArXiv MCP
            "Casual Conversation and General Query": "web_search", # Label 1 → SearXNG 
            "Hybrid Research Query": "hybrid_search",            # Label 2 → Both
            "Web Search and Current Information Query": "web_search" # Label 3 → SearXNG
        }
        
        return intent_mapping.get(intent_class, "hybrid_search")  # Default fallback

# Global classifier instance
_intent_classifier = None

def get_intent_classifier(config: Configuration) -> IntentClassifier:
    """Get or create the global intent classifier instance"""
    global _intent_classifier
    
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier(config)
    
    return _intent_classifier

def classify_query_intent(text: str, config: Configuration) -> Dict[str, Any]:
    """Convenience function to classify intent"""
    classifier = get_intent_classifier(config)
    return classifier.classify_intent(text)