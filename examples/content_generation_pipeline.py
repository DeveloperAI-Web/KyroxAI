#!/usr/bin/env python3
"""
Advanced content generation pipeline using Kyrox agents.
Demonstrates multi-agent collaboration for creating, reviewing,
and optimizing content across different formats.
"""

import asyncio
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import torch
from torch import nn
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
from textblob import TextBlob

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

@dataclass
class ContentRequest:
    topic: str
    target_audience: str
    tone: str
    format: str
    length: int
    keywords: List[str]
    style_guide: Dict
    deadline: datetime

@dataclass
class ContentVersion:
    content: str
    version: int
    timestamp: datetime
    author: str
    metrics: Dict
    feedback: List[str]

class ContentScore(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, embeddings):
        return self.scorer(embeddings)

class ResearchAgent:
    def __init__(self, name: str):
        self.name = name
        self.research_cache = {}
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
    async def research_topic(self, topic: str, keywords: List[str]) -> Dict:
        # Simulate web research
        research_data = {
            "main_points": [
                "Advanced AI systems are revolutionizing content creation",
                "Multi-agent systems enable parallel processing",
                "Natural language understanding is key to quality content"
            ],
            "key_statistics": {
                "market_size": "$5B",
                "growth_rate": "25%",
                "adoption_rate": "65%"
            },
            "expert_quotes": [
                "AI agents are transforming how we create and consume content",
                "Collaborative AI systems show promising results in content generation"
            ]
        }
        
        # Summarize research findings
        combined_text = " ".join([
            *research_data["main_points"],
            *research_data["expert_quotes"]
        ])
        
        summary = self.summarizer(combined_text, max_length=150, min_length=50)[0]["summary_text"]
        research_data["summary"] = summary
        
        return research_data

class ContentGenerator:
    def __init__(self, name: str):
        self.name = name
        self.generator = pipeline("text-generation", model="gpt2-medium")
        self.style_embeddings = {}
        
    def generate_content(self, research: Dict, request: ContentRequest) -> str:
        # Analyze style guide
        style_doc = nlp(" ".join([
            request.style_guide.get("tone", ""),
            request.style_guide.get("voice", ""),
            request.style_guide.get("style", "")
        ]))
        
        # Generate initial content
        prompt = self._create_prompt(research, request)
        generated_text = self.generator(prompt, max_length=request.length, num_return_sequences=1)[0]["generated_text"]
        
        # Apply style constraints
        styled_content = self._apply_style(generated_text, request.style_guide)
        
        return styled_content
        
    def _create_prompt(self, research: Dict, request: ContentRequest) -> str:
        return f"""Topic: {request.topic}
        Audience: {request.target_audience}
        Tone: {request.tone}
        Key Points: {research['summary']}
        Keywords: {', '.join(request.keywords)}
        """
        
    def _apply_style(self, content: str, style_guide: Dict) -> str:
        doc = nlp(content)
        sentences = sent_tokenize(content)
        
        # Apply tone adjustments
        if style_guide.get("tone") == "professional":
            sentences = [self._make_professional(sent) for sent in sentences]
        elif style_guide.get("tone") == "casual":
            sentences = [self._make_casual(sent) for sent in sentences]
            
        return " ".join(sentences)
        
    def _make_professional(self, sentence: str) -> str:
        blob = TextBlob(sentence)
        words = blob.words
        
        # Replace casual words with more formal alternatives
        for i, word in enumerate(words):
            synsets = wordnet.synsets(word)
            if synsets:
                formal_alternatives = [lemma.name() for synset in synsets 
                                    for lemma in synset.lemmas()
                                    if lemma.name() != word]
                if formal_alternatives:
                    words[i] = max(formal_alternatives, key=len)
                    
        return " ".join(words)
        
    def _make_casual(self, sentence: str) -> str:
        # Simplify sentence structure
        doc = nlp(sentence)
        simplified = []
        
        for token in doc:
            if token.dep_ in ["nsubj", "ROOT", "dobj"]:
                simplified.append(token.text)
                
        return " ".join(simplified) if simplified else sentence

class ContentReviewer:
    def __init__(self, name: str):
        self.name = name
        self.quality_model = ContentScore(768)  # Using BERT embedding size
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
    async def review_content(self, content: str, request: ContentRequest) -> Dict:
        # Perform multiple analysis passes
        sentiment_scores = self._analyze_sentiment(content)
        readability_scores = self._analyze_readability(content)
        keyword_scores = self._analyze_keywords(content, request.keywords)
        tone_match = self._analyze_tone(content, request.tone)
        
        # Generate detailed feedback
        feedback = []
        if sentiment_scores["negative"] > 0.3:
            feedback.append("Content may be too negative for target audience")
        if readability_scores["grade_level"] > 12:
            feedback.append("Content may be too complex for general audience")
        if keyword_scores["density"] < 0.01:
            feedback.append("Keyword density is too low")
            
        return {
            "sentiment": sentiment_scores,
            "readability": readability_scores,
            "keyword_metrics": keyword_scores,
            "tone_match": tone_match,
            "feedback": feedback
        }
        
    def _analyze_sentiment(self, content: str) -> Dict:
        sentences = sent_tokenize(content)
        sentiments = self.sentiment_analyzer(sentences)
        
        positive_count = sum(1 for s in sentiments if s["label"] == "POSITIVE")
        negative_count = len(sentiments) - positive_count
        
        return {
            "positive": positive_count / len(sentiments),
            "negative": negative_count / len(sentiments)
        }
        
    def _analyze_readability(self, content: str) -> Dict:
        doc = nlp(content)
        words = len([token for token in doc if not token.is_punct])
        sentences = len(list(doc.sents))
        syllables = sum(self._count_syllables(word.text) for word in doc)
        
        # Calculate Flesch-Kincaid Grade Level
        if sentences == 0:
            grade_level = 0
        else:
            grade_level = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
            
        return {
            "grade_level": grade_level,
            "avg_sentence_length": words / sentences if sentences > 0 else 0,
            "avg_syllables_per_word": syllables / words if words > 0 else 0
        }
        
    def _count_syllables(self, word: str) -> int:
        return len(
            [syl for syl in TextBlob(word).words[0].split('-')]
        )
        
    def _analyze_keywords(self, content: str, keywords: List[str]) -> Dict:
        doc = nlp(content.lower())
        content_words = set(token.text for token in doc if not token.is_stop)
        
        keyword_matches = sum(1 for keyword in keywords 
                            if keyword.lower() in content_words)
                            
        return {
            "density": keyword_matches / len(content_words) if content_words else 0,
            "matches": keyword_matches,
            "missing_keywords": [k for k in keywords 
                               if k.lower() not in content_words]
        }
        
    def _analyze_tone(self, content: str, target_tone: str) -> float:
        doc = nlp(content)
        content_features = self._extract_linguistic_features(doc)
        
        # Compare with target tone features (pre-defined)
        tone_features = {
            "professional": {"formality": 0.8, "complexity": 0.7},
            "casual": {"formality": 0.3, "complexity": 0.4},
            "academic": {"formality": 0.9, "complexity": 0.9}
        }
        
        target = tone_features.get(target_tone, {"formality": 0.5, "complexity": 0.5})
        
        return self._calculate_tone_similarity(content_features, target)
        
    def _extract_linguistic_features(self, doc) -> Dict:
        return {
            "formality": self._calculate_formality(doc),
            "complexity": self._calculate_complexity(doc)
        }
        
    def _calculate_formality(self, doc) -> float:
        formal_pos = {"NOUN", "ADJ", "ADP"}
        informal_pos = {"INTJ", "PART"}
        
        formal_count = sum(1 for token in doc if token.pos_ in formal_pos)
        informal_count = sum(1 for token in doc if token.pos_ in informal_pos)
        
        total = len([token for token in doc if token.pos_ in formal_pos | informal_pos])
        return formal_count / total if total > 0 else 0.5
        
    def _calculate_complexity(self, doc) -> float:
        avg_word_length = np.mean([len(token.text) for token in doc])
        avg_sentence_length = np.mean([len(sent) for sent in doc.sents])
        
        # Normalize to 0-1 range
        word_length_score = min(avg_word_length / 10, 1)
        sentence_length_score = min(avg_sentence_length / 30, 1)
        
        return (word_length_score + sentence_length_score) / 2
        
    def _calculate_tone_similarity(self, features: Dict, target: Dict) -> float:
        feature_vector = np.array([features["formality"], features["complexity"]])
        target_vector = np.array([target["formality"], target["complexity"]])
        
        return float(cosine_similarity(
            feature_vector.reshape(1, -1),
            target_vector.reshape(1, -1)
        )[0][0])

async def main():
    # Initialize agents
    researcher = ResearchAgent("researcher")
    generator = ContentGenerator("generator")
    reviewer = ContentReviewer("reviewer")
    
    # Create content request
    request = ContentRequest(
        topic="AI Agent Networks",
        target_audience="Technical professionals",
        tone="professional",
        format="article",
        length=1000,
        keywords=["AI", "agents", "collaboration", "distributed systems"],
        style_guide={
            "tone": "professional",
            "voice": "active",
            "style": "technical but accessible"
        },
        deadline=datetime.now()
    )
    
    # Execute content generation pipeline
    print("Starting content generation pipeline...")
    
    # Step 1: Research
    print("\nResearching topic...")
    research_data = await researcher.research_topic(request.topic, request.keywords)
    
    # Step 2: Generate content
    print("\nGenerating content...")
    content = generator.generate_content(research_data, request)
    
    # Step 3: Review content
    print("\nReviewing content...")
    review_results = await reviewer.review_content(content, request)
    
    # Print results
    print("\nGenerated Content:")
    print("-" * 50)
    print(content[:500] + "...")
    print("\nReview Results:")
    print("-" * 50)
    print(json.dumps(review_results, indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 