"""
AI Chatbot Assistant for Map Analysis
Provides natural language interface for querying satellite imagery analysis
"""

import numpy as np
import re
from lulc_classifier import LULC_CLASSES, analyze_spatial_distribution


class MapChatbot:
    """
    Rule-based chatbot for map analysis.
    Can be extended with Gemini API for advanced queries.
    """
    
    def __init__(self, use_gemini=False, api_key=None):
        """
        Initialize chatbot.
        
        Args:
            use_gemini: bool, whether to use Gemini API
            api_key: str, Gemini API key (optional)
        """
        self.use_gemini = use_gemini
        self.api_key = api_key
        self.gemini_model = None
        
        if use_gemini and api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
            except:
                self.use_gemini = False
    
    def parse_query(self, user_input):
        """
        Extract intent and entities from user query.
        
        Args:
            user_input: str, user's question
        
        Returns:
            dict with 'intent' and 'entities'
        """
        user_input = user_input.lower()
        
        # Detect intent
        intent = 'unknown'
        entities = {'class': None, 'metric': None}
        
        # Percentage/amount queries
        if any(word in user_input for word in ['how much', 'percentage', '%', 'amount', 'evlo']):
            intent = 'percentage'
        
        # Location queries
        elif any(word in user_input for word in ['where', 'location', 'enga', 'idathula']):
            intent = 'location'
        
        # Comparison queries
        elif any(word in user_input for word in ['more', 'less', 'compare', 'vs', 'or']):
            intent = 'comparison'
        
        # General info
        elif any(word in user_input for word in ['what', 'tell me', 'describe', 'enna']):
            intent = 'info'
        
        # Detect land cover class
        for class_id, class_info in LULC_CLASSES.items():
            class_name = class_info['name'].lower()
            # Also check Tamil/Tanglish keywords
            keywords = {
                'water': ['water', 'thanni', 'river', 'lake', 'ocean'],
                'forest': ['forest', 'kaadu', 'tree', 'jungle'],
                'urban': ['urban', 'veedu', 'city', 'building', 'road'],
                'barren': ['barren', 'vettaveli', 'bare', 'desert'],
                'vegetation': ['vegetation', 'pachai', 'green', 'crop', 'farm']
            }
            
            if class_name in keywords:
                if any(kw in user_input for kw in keywords[class_name]):
                    entities['class'] = class_id
                    break
        
        return {'intent': intent, 'entities': entities}
    
    def analyze_statistics(self, lulc_data):
        """
        Compute statistics from LULC data.
        
        Args:
            lulc_data: dict from classify_and_visualize()
        
        Returns:
            dict with various statistics
        """
        percentages = lulc_data['percentages']
        mask = lulc_data['mask']
        
        stats = {
            'percentages': percentages,
            'dominant_class': max(percentages, key=percentages.get),
            'locations': {}
        }
        
        # Calculate spatial distribution for each class
        for class_id, class_info in LULC_CLASSES.items():
            class_name = class_info['name']
            location = analyze_spatial_distribution(mask, class_id)
            stats['locations'][class_name] = location
        
        return stats
    
    def generate_response(self, query, lulc_data):
        """
        Generate response to user query.
        
        Args:
            query: str, user's question
            lulc_data: dict from classify_and_visualize()
        
        Returns:
            str, response message
        """
        parsed = self.parse_query(query)
        intent = parsed['intent']
        class_id = parsed['entities']['class']
        
        stats = self.analyze_statistics(lulc_data)
        percentages = stats['percentages']
        
        # Generate response based on intent
        if intent == 'percentage':
            if class_id is not None:
                class_name = LULC_CLASSES[class_id]['name']
                pct = percentages[class_name]
                location = stats['locations'][class_name]
                return f"📊 **{class_name}** covers **{pct:.1f}%** of this area. {location}."
            else:
                # Show all percentages
                response = "📊 **Land Cover Distribution:**\n\n"
                for name, pct in percentages.items():
                    if pct > 0.5:
                        emoji = LULC_CLASSES[[k for k, v in LULC_CLASSES.items() if v['name'] == name][0]]['label'].split()[0]
                        response += f"- {emoji} **{name}**: {pct:.1f}%\n"
                return response
        
        elif intent == 'location':
            if class_id is not None:
                class_name = LULC_CLASSES[class_id]['name']
                location = stats['locations'][class_name]
                pct = percentages[class_name]
                return f"📍 **{class_name}** ({pct:.1f}%) is located: **{location}**"
            else:
                return "🤔 Please specify which land cover type you're asking about (water, forest, urban, barren, or vegetation)."
        
        elif intent == 'comparison':
            # Find two classes mentioned
            classes_found = []
            for class_id, class_info in LULC_CLASSES.items():
                if class_info['name'].lower() in query.lower():
                    classes_found.append(class_info['name'])
            
            if len(classes_found) >= 2:
                class1, class2 = classes_found[0], classes_found[1]
                pct1, pct2 = percentages[class1], percentages[class2]
                
                if pct1 > pct2:
                    diff = pct1 - pct2
                    return f"📊 **{class1}** ({pct1:.1f}%) is more prevalent than **{class2}** ({pct2:.1f}%) by **{diff:.1f}%**."
                elif pct2 > pct1:
                    diff = pct2 - pct1
                    return f"📊 **{class2}** ({pct2:.1f}%) is more prevalent than **{class1}** ({pct1:.1f}%) by **{diff:.1f}%**."
                else:
                    return f"📊 **{class1}** and **{class2}** have similar coverage (~{pct1:.1f}%)."
            else:
                # General comparison
                sorted_classes = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
                top_class, top_pct = sorted_classes[0]
                return f"🏆 **{top_class}** is the dominant land cover type at **{top_pct:.1f}%**."
        
        elif intent == 'info':
            dominant = stats['dominant_class']
            dominant_pct = percentages[dominant]
            
            response = f"🗺️ **Map Analysis Summary:**\n\n"
            response += f"🏆 Dominant type: **{dominant}** ({dominant_pct:.1f}%)\n\n"
            response += "**Distribution:**\n"
            
            for name, pct in sorted(percentages.items(), key=lambda x: x[1], reverse=True):
                if pct > 1.0:
                    emoji = LULC_CLASSES[[k for k, v in LULC_CLASSES.items() if v['name'] == name][0]]['label'].split()[0]
                    response += f"- {emoji} {name}: {pct:.1f}%\n"
            
            return response
        
        else:
            return "🤔 I can help you analyze this map! Try asking:\n" + \
                   "- 'How much water is in this area?'\n" + \
                   "- 'Where is the urban area located?'\n" + \
                   "- 'Is there more forest or vegetation?'\n" + \
                   "- 'Tell me about this map'"
    
    def chat(self, query, lulc_data):
        """
        Main chat interface.
        
        Args:
            query: str, user's question
            lulc_data: dict from classify_and_visualize()
        
        Returns:
            str, response
        """
        if self.use_gemini and self.gemini_model:
            # Use Gemini for advanced responses
            try:
                stats = self.analyze_statistics(lulc_data)
                context = f"Land cover analysis: {stats['percentages']}"
                prompt = f"{context}\n\nUser question: {query}\n\nProvide a helpful response."
                response = self.gemini_model.generate_content(prompt)
                return response.text
            except:
                # Fallback to rule-based
                return self.generate_response(query, lulc_data)
        else:
            # Use rule-based
            return self.generate_response(query, lulc_data)
