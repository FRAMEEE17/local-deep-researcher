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
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        
        # Load label encoder (saved with torch.save) - use weights_only=False for sklearn objects
        self.label_encoder = torch.load(config.label_encoder_path, map_location='cpu', weights_only=False)
        
        # Load model (4 classes from your training) - use weights_only=True for model weights
        self.model = CustomXLMRobertaModel(num_labels=4)
        
        # Load state dict with strict=False to handle transformer version differences
        state_dict = torch.load(config.intent_model_path, map_location='cpu', weights_only=True)
        
        # Remove problematic keys that may exist in older transformer versions
        keys_to_remove = [k for k in state_dict.keys() if 'position_ids' in k]
        for key in keys_to_remove:
            del state_dict[key]
            
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Intent classifier loaded on {self.device}")
        print(f"Available intents: {list(self.label_encoder.classes_)}")
    
    def classify_intent(self, text: str) -> Dict[str, Any]:
        start_time = time.time()
        
        # Preprocess: Detect ArXiv patterns and enhance text for better classification
        processed_text = self._preprocess_for_classification(text)
        
        # Tokenize 
        inputs = self.tokenizer(
            processed_text,
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
        # Create all_probabilities dictionary
        all_probabilities = {
            self.label_encoder.inverse_transform([i])[0]: prob.item()
            for i, prob in enumerate(probabilities[0])
        }
        # Decode using your trained label encoder
        intent_class = self.label_encoder.inverse_transform([prediction])[0]
        
        # Map to routing strategy based on your trained classes
        routing_strategy = self._map_intent_to_strategy(intent_class)
        
        processing_time = time.time() - start_time
        confidence_analysis = self.interpret_model_confidence(all_probabilities, routing_strategy)
        return {
            "intent": intent_class,
            "confidence": confidence_analysis["calibrated_confidence"],  # Use calibrated
            "raw_confidence": confidence,  # Keep original
            "routing_strategy": routing_strategy,
            "confidence_analysis": confidence_analysis,
            "processing_time": processing_time,
            "all_probabilities": all_probabilities,
            "model_interpretation": confidence_analysis["interpretation"]
        }
    
    def _map_intent_to_strategy(self, intent_class: str) -> str:
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
    
    def _preprocess_for_classification(self, text: str) -> str:
        import re
        
        processed_text = text.strip()
        context_signals = []
        
        # ARXIV DETECTION (Improved patterns)
        arxiv_patterns = [
            r'https?://arxiv\.org/(?:abs|html|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)',
            r'arxiv\.org/(?:abs|html|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)',
            r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)',
            r'\b(\d{4}\.\d{4,5}(?:v\d+)?)\b',  # Standalone ArXiv IDs
            r'paper\s+(\d{4}\.\d{4,5}(?:v\d+)?)',  # "paper 1706.03762"
        ]
        
        arxiv_found = False
        for pattern in arxiv_patterns:
            if re.search(pattern, processed_text, re.IGNORECASE):
                arxiv_found = True
                context_signals.append("arxiv_paper_reference")
                break
        
        # ACADEMIC SIGNAL ENHANCEMENT
        academic_indicators = {
            'paper_verbs': r'\b(explain|analyze|summarize|review|discuss)\s+(?:the\s+)?paper\b',
            'research_verbs': r'\b(study|research|investigate|examine|explore)\b',
            'academic_terms': r'\b(abstract|methodology|results|conclusion|findings|experiment)\b',
            'citation_style': r'\b(et al\.|Figure \d+|Table \d+|Section \d+)\b',
            'venue_mentions': r'\b(conference|journal|proceedings|publication|acm|ieee)\b'
        }
        
        for signal_type, pattern in academic_indicators.items():
            if re.search(pattern, processed_text, re.IGNORECASE):
                context_signals.append(signal_type)
        
        # TEMPORAL SIGNAL ENHANCEMENT  
        temporal_indicators = {
            'recent_time': r'\b(recent|latest|current|new|2024|2025|today|now)\b',
            'temporal_verbs': r'\b(happening|trending|breaking|updated)\b',
            'news_terms': r'\b(news|article|report|announcement|release)\b'
        }
        
        for signal_type, pattern in temporal_indicators.items():
            if re.search(pattern, processed_text, re.IGNORECASE):
                context_signals.append(signal_type)
        
        # QUESTION TYPE DETECTION
        question_patterns = {
            'explanation': r'\b(what is|how does|why|explain|describe)\b',
            'comparison': r'\b(compare|vs|versus|difference|better)\b',
            'application': r'\b(how to|implement|use|apply|practical)\b',
            'overview': r'\b(overview|survey|review|introduction)\b'
        }
        
        for q_type, pattern in question_patterns.items():
            if re.search(pattern, processed_text, re.IGNORECASE):
                context_signals.append(f"question_{q_type}")
        
        # CONTEXT ENHANCEMENT
        if arxiv_found:
            processed_text += " [ACADEMIC_PAPER_REFERENCE]"
        
        if any("recent" in signal or "temporal" in signal for signal in context_signals):
            processed_text += " [TEMPORAL_QUERY]"
        
        if any("question" in signal for signal in context_signals):
            question_types = [s.replace("question_", "") for s in context_signals if "question_" in s]
            processed_text += f" [QUESTION_TYPE:{','.join(question_types)}]"
        
        # DOMAIN DETECTION
        domain_patterns = {
            'ai_ml': r'\b(machine learning|deep learning|neural|transformer|llm|gpt|bert)\b',
            'computer_science': r'\b(algorithm|data structure|programming|software|computer)\b',
            'physics': r'\b(quantum|physics|particle|theory|relativity)\b',
            'biology': r'\b(biology|genetics|protein|dna|medical|clinical)\b',
            'mathematics': r'\b(theorem|proof|equation|mathematical|statistics)\b'
        }
        
        detected_domains = []
        for domain, pattern in domain_patterns.items():
            if re.search(pattern, processed_text, re.IGNORECASE):
                detected_domains.append(domain)
        
        if detected_domains:
            processed_text += f" [DOMAINS:{','.join(detected_domains)}]"
        return processed_text
    def interpret_model_confidence(self, probabilities: dict, prediction: str) -> Dict[str, Any]:
        """Interpret model confidence with uncertainty quantification."""
        import math
        
        # Calculate prediction certainty metrics
        max_prob = max(probabilities.values())
        second_max = sorted(probabilities.values(), reverse=True)[1] if len(probabilities) > 1 else 0
        
        confidence_metrics = {
            "raw_confidence": max_prob,
            "certainty_gap": max_prob - second_max,  # Gap between top predictions
            "entropy": -sum(p * math.log2(p + 1e-10) for p in probabilities.values()),
            "prediction_dominance": max_prob / sum(probabilities.values()) if sum(probabilities.values()) > 0 else 0
        }
        
        # Confidence interpretation
        if max_prob > 0.9 and confidence_metrics["certainty_gap"] > 0.3:
            confidence_level = "very_high"
            calibrated_confidence = max_prob * 0.95
        elif max_prob > 0.8 and confidence_metrics["certainty_gap"] > 0.2:
            confidence_level = "high" 
            calibrated_confidence = max_prob * 0.9
        elif max_prob > 0.6 and confidence_metrics["certainty_gap"] > 0.1:
            confidence_level = "moderate"
            calibrated_confidence = max_prob * 0.85
        else:
            confidence_level = "low"
            calibrated_confidence = max_prob * 0.8
        
        # Strategy-specific adjustments
        strategy_reliability = {
            "arxiv_search": 0.95,    # Academic classification is usually reliable
            "hybrid_search": 0.85,   # Hybrid requires more certainty  
            "web_search": 0.9        # Web is broad category
        }
        
        strategy_multiplier = strategy_reliability.get(prediction, 0.85)
        final_confidence = calibrated_confidence * strategy_multiplier
        final_confidence = max(0.1, min(0.99, final_confidence))  # Bound confidence
        
        return {
            "calibrated_confidence": final_confidence,
            "confidence_level": confidence_level,
            "confidence_metrics": confidence_metrics,
            "interpretation": f"{confidence_level} confidence ({final_confidence:.3f})",
            "uncertainty_flags": [
                "low_certainty_gap" if confidence_metrics["certainty_gap"] < 0.15 else None,
                "high_entropy" if confidence_metrics["entropy"] > 1.5 else None,
                "weak_dominance" if confidence_metrics["prediction_dominance"] < 0.6 else None
            ]
        }
# Global classifier instance
_intent_classifier = None

def get_intent_classifier(config: Configuration) -> IntentClassifier:
    global _intent_classifier
    
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier(config)
    
    return _intent_classifier

def classify_query_intent(text: str, config: Configuration) -> Dict[str, Any]:
    classifier = get_intent_classifier(config)
    return classifier.classify_intent(text)