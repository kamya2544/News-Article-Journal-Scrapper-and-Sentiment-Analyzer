from abc import ABC, abstractmethod
import heapq
import os
from typing import Any, Dict, List, Tuple

import easyocr
import numpy as np
import spacy
import torch
from PIL import Image
from transformers import pipeline
from loguru import logger

from sentiment_analysis.core import settings, SentimentLabel

# Base Provider Interfaces

class SentimentProvider(ABC):
    @abstractmethod
    def analyze_overall(self, text: str) -> Tuple[SentimentLabel, float, str]:
        """Returns (Label, Confidence, Reasoning)."""
        pass

    @abstractmethod
    def analyze_aspect(self, text: str, aspect: str, sentences: List[str]) -> Tuple[SentimentLabel, float, str]:
        """Returns (Label, Confidence, Reason/Supporting Sentences)."""
        pass


class NERProvider(ABC):
    @abstractmethod
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Returns a list of dicts with keys 'text' and 'label'."""
        pass


class OCRProvider(ABC):
    @abstractmethod
    def run_ocr(self, image: Image.Image) -> Dict[str, Any]:
        """Returns dict with keys: success (bool), text (str), confidence (float)."""
        pass


class SummarizerProvider(ABC):
    @abstractmethod
    def summarize(self, text: str) -> str:
        """Returns a concise summary of the article."""
        pass


# Concrete Implementations

class FinBERTSentimentProvider(SentimentProvider):
    def __init__(self):
        self._pipeline = None
        self.model_name = settings.MODEL_NAME

    def _get_pipeline(self):
        if self._pipeline is None:
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Loading HF sentiment analysis pipeline for model {self.model_name} on device {device}")
            try:
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=device
                )
            except Exception as e:
                logger.error(f"Failed to load FinBERT pipeline: {str(e)}. Falling back to simple heuristic.")
                self._pipeline = "fallback"
        return self._pipeline

    def _fallback_sentiment(self, text: str) -> Tuple[SentimentLabel, float, str]:
        # Clean basic heuristic for fallback
        text_lower = text.lower()
        pos_words = ["increase", "growth", "subsidy", "profit", "gain", "announce", "positive", "expand", "boost"]
        neg_words = ["decrease", "decline", "loss", "headwind", "cut", "negative", "tighten", "drop", "regulation"]
        
        pos_count = sum(1 for w in pos_words if w in text_lower)
        neg_count = sum(1 for w in neg_words if w in text_lower)
        
        confidence = 0.70
        if pos_count > neg_count:
            return SentimentLabel.POSITIVE, confidence, f"Fallback Heuristic: Found {pos_count} positive keywords vs {neg_count} negative keywords."
        elif neg_count > pos_count:
            return SentimentLabel.NEGATIVE, confidence, f"Fallback Heuristic: Found {neg_count} negative keywords vs {pos_count} positive keywords."
        else:
            return SentimentLabel.NEUTRAL, 0.60, "Fallback Heuristic: Balanced or insufficient sentiment keywords."

    def analyze_overall(self, text: str) -> Tuple[SentimentLabel, float, str]:
        pipeline = self._get_pipeline()
        if pipeline == "fallback" or pipeline is None:
            return self._fallback_sentiment(text)
        
        try:
            # HF Pipeline can complain about very long text, chunk it if necessary
            truncated_text = text[:1500]  # FinBERT limit around 512 tokens
            result = pipeline(truncated_text)[0]
            label_map = {
                "positive": SentimentLabel.POSITIVE,
                "neutral": SentimentLabel.NEUTRAL,
                "negative": SentimentLabel.NEGATIVE,
                "LABEL_0": SentimentLabel.NEGATIVE,
                "LABEL_1": SentimentLabel.NEUTRAL,
                "LABEL_2": SentimentLabel.POSITIVE,
            }
            raw_label = result["label"].lower()
            label = label_map.get(raw_label, SentimentLabel.NEUTRAL)
            confidence = float(result["score"])
            
            reasoning = f"FinBERT predicted '{label.value}' sentiment with {confidence:.2%} confidence based on content analysis."
            return label, confidence, reasoning
        except Exception as e:
            logger.error(f"Inference error in FinBERT: {str(e)}")
            return self._fallback_sentiment(text)

    def analyze_aspect(self, text: str, aspect: str, sentences: List[str]) -> Tuple[SentimentLabel, float, str]:
        # Aspect-Based Sentiment Analysis using sentence proximity method
        pipeline = self._get_pipeline()
        
        # Get aspect keyword synonyms
        aspect_synonyms = {
            "biofuels": ["biofuel", "ethanol", "biodiesel", "biomass", "neste"],
            "petrochemicals": ["petrochemical", "crude", "oil", "plastic", "ethylene", "propylene", "refinery"],
            "hydrogen": ["hydrogen", "h2", "electrolyzer"],
            "saf": ["saf", "aviation fuel", "sustainable aviation"],
            "renewable energy": ["renewable", "solar", "wind", "geothermal", "clean energy"],
            "carbon capture": ["carbon capture", "ccus", "co2 capture", "sequestration", "emissions"]
        }
        
        keywords = aspect_synonyms.get(aspect.lower(), [aspect.lower()])
        matching_sentences = []
        
        for s in sentences:
            s_lower = s.lower()
            if any(kw in s_lower for kw in keywords):
                matching_sentences.append(s)

        if not matching_sentences:
            return SentimentLabel.NEUTRAL, 0.5, f"No direct evidence or sentences found mentioning aspect '{aspect}'."

        if pipeline == "fallback" or pipeline is None:
            # Combine matching sentences and run fallback
            combined = " ".join(matching_sentences)
            label, conf, reason = self._fallback_sentiment(combined)
            return label, conf, f"Aspect classified using fallback. Evidence: {matching_sentences[0]}"

        try:
            scores = {SentimentLabel.POSITIVE: 0.0, SentimentLabel.NEUTRAL: 0.0, SentimentLabel.NEGATIVE: 0.0}
            counts = {SentimentLabel.POSITIVE: 0, SentimentLabel.NEUTRAL: 0, SentimentLabel.NEGATIVE: 0}
            
            label_map = {
                "positive": SentimentLabel.POSITIVE,
                "neutral": SentimentLabel.NEUTRAL,
                "negative": SentimentLabel.NEGATIVE,
                "LABEL_0": SentimentLabel.NEGATIVE,
                "LABEL_1": SentimentLabel.NEUTRAL,
                "LABEL_2": SentimentLabel.POSITIVE,
            }

            for s in matching_sentences[:5]:  # Limit to top 5 sentences
                res = pipeline(s[:1000])[0]
                lbl = label_map.get(res["label"].lower(), SentimentLabel.NEUTRAL)
                scores[lbl] += res["score"]
                counts[lbl] += 1
                
            # Determine dominant label
            max_label = SentimentLabel.NEUTRAL
            max_val = -1
            for lbl in SentimentLabel:
                if counts[lbl] > max_val:
                    max_val = counts[lbl]
                    max_label = lbl
                elif counts[lbl] == max_val and max_val > 0:
                    if scores[lbl] > scores[max_label]:
                        max_label = lbl
            
            total_count = sum(counts.values())
            avg_confidence = scores[max_label] / counts[max_label] if counts[max_label] > 0 else 0.5
            
            evidence_str = matching_sentences[0] if matching_sentences else ""
            reason = f"Classified as {max_label.value} based on {counts[max_label]} supporting sentence(s). Key evidence: \"{evidence_str}\""
            
            return max_label, avg_confidence, reason
        except Exception as e:
            logger.error(f"Aspect inference error for '{aspect}': {str(e)}")
            return SentimentLabel.NEUTRAL, 0.5, f"Error analyzing aspect. Defaulted to Neutral."


class SpacyNERProvider(NERProvider):
    def __init__(self):
        self._nlp = None
        self.model_name = settings.NER_MODEL

    def _get_nlp(self):
        if self._nlp is None:
            logger.info(f"Loading spaCy model {self.model_name}")
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError:
                logger.warning(f"spaCy model {self.model_name} not found. Attempting to download...")
                spacy.cli.download(self.model_name)
                self._nlp = spacy.load(self.model_name)
        return self._nlp

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        nlp = self._get_nlp()
        doc = nlp(text)
        entities = []
        seen = set()
        
        # Mapping spaCy entity labels to client-facing entity tags
        label_mapping = {
            "ORG": "Organization",
            "GPE": "Country",
            "LOC": "Location",
            "PRODUCT": "Product",
            "MONEY": "Financial",
            "LAW": "Policy"
        }
        
        # Energy sector entity keywords matching rule-based fallback inside spaCy pipeline
        chemical_keywords = ["ethanol", "crude", "methanol", "biodiesel", "hydrogen", "ammonia", "ethylene", "naphtha", "benzene", "co2"]
        tech_keywords = ["electrolyzer", "carbon capture", "ccus", "refinery", "solar panel", "wind turbine", "catalyst"]

        # Run model entities
        for ent in doc.ents:
            cleaned_text = ent.text.strip().replace("\n", " ")
            if len(cleaned_text) < 2:
                continue
            entity_key = (cleaned_text.lower(), ent.label_)
            if entity_key not in seen:
                seen.add(entity_key)
                mapped_label = label_mapping.get(ent.label_, ent.label_)
                entities.append({
                    "text": cleaned_text,
                    "label": mapped_label
                })

        # Add rule-based logic for Chemical & Product detection if not found by spaCy model
        doc_lower = text.lower()
        for chem in chemical_keywords:
            if chem in doc_lower and not any(chem in e["text"].lower() for e in entities):
                # Simple capitalization
                entities.append({"text": chem.capitalize(), "label": "Chemical"})
        for tech in tech_keywords:
            if tech in doc_lower and not any(tech in e["text"].lower() for e in entities):
                entities.append({"text": tech.capitalize(), "label": "Technology"})

        return entities


class EasyOCROCRProvider(OCRProvider):
    def __init__(self):
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            if settings.ENABLE_OCR:
                logger.info("Initializing EasyOCR Reader for English...")
                try:
                    self._reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                except Exception as e:
                    logger.error(f"Failed to initialize EasyOCR: {str(e)}")
                    self._reader = "failed"
            else:
                self._reader = "disabled"
        return self._reader

    def run_ocr(self, image: Image.Image) -> Dict[str, Any]:
        reader = self._get_reader()
        if reader == "disabled":
            return {"success": False, "text": "", "confidence": 0.0}
        if reader == "failed" or reader is None:
            return {"success": False, "text": "", "confidence": 0.0}

        try:
            # We convert the PIL Image to a numpy array because EasyOCR processes arrays.
            img_np = np.array(image.convert("RGB"))
            results = reader.readtext(img_np)
            if not results:
                return {"success": True, "text": "", "confidence": 1.0}
            
            texts = []
            confidences = []
            for bbox, text, conf in results:
                if conf > 0.3:  # We ignore low-confidence results to prevent garbled text.
                    texts.append(text)
                    confidences.append(conf)
            
            combined_text = " ".join(texts)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                "success": True,
                "text": combined_text,
                "confidence": avg_conf
            }
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return {"success": False, "text": "", "confidence": 0.0}


class SpacySummarizerProvider(SummarizerProvider):
    def __init__(self):
        self._nlp = None
        self.model_name = settings.NER_MODEL

    def _get_nlp(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError:
                spacy.cli.download(self.model_name)
                self._nlp = spacy.load(self.model_name)
        return self._nlp

    def summarize(self, text: str) -> str:
        # Perform Extractive Summarization using word frequency
        nlp = self._get_nlp()
        doc = nlp(text)
        
        sentences = [sent for sent in doc.sents]
        if len(sentences) <= 3:
            return text.strip()

        # Word frequency map (excluding stop words and punctuation)
        word_frequencies = {}
        for word in doc:
            if not word.is_stop and not word.is_punct and not word.is_space:
                word_frequencies[word.text.lower()] = word_frequencies.get(word.text.lower(), 0) + 1

        if not word_frequencies:
            return " ".join([s.text.strip() for s in sentences[:3]])

        max_frequency = max(word_frequencies.values())
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / max_frequency

        # Sentence scoring
        sentence_scores = {}
        for sent in sentences:
            for word in sent:
                if word.text.lower() in word_frequencies:
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word.text.lower()]

        # We select the top 3 sentences with the highest frequency scores and sort them chronologically.
        top_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)
        ordered_sentences = sorted(top_sentences, key=lambda s: s.start)
        
        summary = " ".join([s.text.strip().replace("\n", " ") for s in ordered_sentences])
        return summary


# Provider Factory

class ProviderFactory:
    def __init__(self):
        self._sentiment_provider = None
        self._ner_provider = None
        self._ocr_provider = None
        self._summarizer_provider = None

    def get_sentiment_provider(self) -> SentimentProvider:
        if self._sentiment_provider is None:
            self._sentiment_provider = FinBERTSentimentProvider()
        return self._sentiment_provider

    def get_ner_provider(self) -> NERProvider:
        if self._ner_provider is None:
            self._ner_provider = SpacyNERProvider()
        return self._ner_provider

    def get_ocr_provider(self) -> OCRProvider:
        if self._ocr_provider is None:
            self._ocr_provider = EasyOCROCRProvider()
        return self._ocr_provider

    def get_summarizer_provider(self) -> SummarizerProvider:
        if self._summarizer_provider is None:
            self._summarizer_provider = SpacySummarizerProvider()
        return self._summarizer_provider

_factory = ProviderFactory()

def get_provider_factory() -> ProviderFactory:
    return _factory
