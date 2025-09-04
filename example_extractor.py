# example_extractor.py
import os
import json
import time
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

class GSEBExampleExtractor:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Gemini
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # API call counter
        self.api_calls = {"gemini_api": 0}
        
        print("📚 Initialized GSEB Example Extractor")
        print(f"🔑 Gemini API Key: {'✅ Loaded' if gemini_api_key else '❌ Missing'}")

    def load_processed_json(self, json_file_path):
        """Load processed JSON from main.py output"""
        print(f"\n📂 Loading processed JSON: {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 Loaded data:")
        print(f"  📄 Total pages: {len(data.get('pages', []))}")
        print(f"  📚 Chapter: {data.get('metadata', {}).get('source_pdf', 'Unknown')}")
        
        return data


    def extract_examples_from_chapter(self, json_data):
        """Extract examples with complete visual descriptions"""
        print("\n" + "="*50)
        print("🔍 EXAMPLE EXTRACTION WITH VISUAL DESCRIPTIONS")
        print("="*50)
        
        pages_data = json_data.get('pages', [])
        chapter_name = self._get_chapter_name(json_data)
        
        all_examples = []
        
        for page in tqdm(pages_data, desc="🔍 Extracting examples"):
            try:
                page_number = page.get('page_number', 0)
                page_text = page.get('text', '')
                
                if len(page_text) < 100:
                    continue
                    
                # Check for example indicators
                example_indicators = ['ઉદાહરણ', 'Example', 'ઉકેલ', 'હલ']
                if not any(indicator in page_text for indicator in example_indicators):
                    continue
                
                # Step 1: Extract examples with AI
                examples = self._extract_examples_with_ai(page_text, chapter_name, page_number)
                
                if examples:
                    # Step 2: Enhance with visual descriptions
                    examples = self._enhance_examples_with_visual_content(examples, page)
                    
                    for example in examples:
                        visuals_count = len(example.get('mentioned_visuals', []))
                        print(f"    📝 Example {example.get('example_number', '')}: {visuals_count} visual references")
                    
                    all_examples.extend(examples)
                
                time.sleep(1)
                
            except Exception as e:
                print(f"  ❌ Error on page {page_number}: {str(e)[:100]}...")
        
        print(f"\n📚 EXTRACTION COMPLETED")
        print(f"🔍 Total examples: {len(all_examples)}")
        
        # Statistics
        total_visuals = sum(len(ex.get('mentioned_visuals', [])) for ex in all_examples)
        visuals_with_descriptions = sum(
            sum(1 for v in ex.get('mentioned_visuals', []) if v.get('full_description') != "વર્ણન ઉપલબ્ધ નથી") 
            for ex in all_examples
        )
        
        print(f"📊 Visual References:")
        print(f"  📝 Total mentioned: {total_visuals}")
        print(f"  📄 With descriptions: {visuals_with_descriptions}")
        
        return all_examples


    def _find_matching_visual_description(self, visual_reference, visual_type, page_images):
        """Find matching image description based on reference and type"""
        import re
        
        # Method 1: Exact reference matching (કોષ્ટક 3.1, આકૃતિ 3.2)
        if visual_reference:
            ref_match = re.search(r'\d+\.\d+', visual_reference)
            if ref_match:
                ref_number = ref_match.group()
                ref_type = visual_reference.split()[0]  # કોષ્ટક, આકૃતિ
                
                for image in page_images:
                    img_desc = image.get('educational_description', '')
                    if ref_number in img_desc and ref_type in img_desc:
                        return img_desc
        
        # Method 2: Type-based matching
        visual_type_lower = visual_type.lower()
        
        type_keywords = {
            'કોષ્ટક': ['કોષ્ટક', 'table'],
            'આકૃતિ': ['આકૃતિ', 'આલેખ', 'graph'],
            'ચિત્ર': ['ચિત્ર', 'diagram', 'figure']
        }
        
        if visual_type_lower in type_keywords:
            keywords = type_keywords[visual_type_lower]
            
            for image in page_images:
                img_desc = image.get('educational_description', '').lower()
                if any(keyword in img_desc for keyword in keywords):
                    return image.get('educational_description', '')
        
        return None

    def _enhance_examples_with_visual_content(self, examples, page_data):
        """Add actual image descriptions to mentioned visual references"""
        
        page_images = page_data.get('images', [])
        if not page_images:
            return examples
        
        for example in examples:
            mentioned_visuals = example.get('mentioned_visuals', [])
            
            for visual in mentioned_visuals:
                visual_ref = visual.get('reference', '')
                visual_type = visual.get('type', '')
                
                # Find matching description from page images
                matching_desc = self._find_matching_visual_description(
                    visual_ref, visual_type, page_images
                )
                
                if matching_desc:
                    visual['full_description'] = matching_desc
                    print(f"    ✅ Found description for {visual_ref}")
                else:
                    visual['full_description'] = "વર્ણન ઉપલબ્ધ નથી"
                    print(f"    ⚠️ No description found for {visual_ref}")
        
        return examples


    def _extract_final_answer_from_text(self, page_text, example_text):
        """Extract the final numerical answer from the solution"""
        import re
        
        # Look for patterns like "∴ x = 8", "જવાબ:", "ઉકેલ:", etc.
        answer_indicators = [
            r'∴\s*([^.]+)',
            r'જવાબ\s*:?\s*([^.]+)',
            r'ઉકેલ\s*:?\s*([^.]+)',
            r'x\s*=\s*\d+.*y\s*=\s*\d+',
            r'\w+\s*=\s*\d+[^.]*'
        ]
        
        for pattern in answer_indicators:
            matches = re.findall(pattern, page_text, re.MULTILINE)
            if matches:
                # Return the most complete numerical answer
                for match in matches:
                    if any(char.isdigit() for char in match):
                        return match.strip()
        
        return "ઉકેલ પૂર્ણ કરવાની જરૂર છે"



    def _extract_examples_with_ai(self, page_text, chapter_name, page_number):
        """Extract examples with example numbers and mentioned visual references"""
        
    
        prompt = f"""
        તમે ધોરણ 10 ગણિતના અધ્યાય "{chapter_name}" ના પાના {page_number} માંથી **ઉદાહરણો** શોધી રહ્યા છો.
        
        દરેક ઉદાહરણ માટે:
        1. **example_number**: "ઉદાહરણ 19" વગેરે
        2. **question**: મૂળ સમસ્યા/પ્રશ્ન
        3. **answer**: અંતિમ જવાબ (આંકડાકીય મૂલ્યો સાથે - જેમ કે x = 8, y = 3)
        4. **explanation**: સંપૂર્ણ ઉકેલની પદ્ધતિ (પગલા દ્વારા)
        
        **મહત્વપૂર્ણ સૂચનાઓ:**
        - "ઉકેલ:", "જવાબ:", "∴" પછી આવતો ભાગ એ અંતિમ જવાબ છે
        - સમીકરણો અને ગણતરીઓ explanation માં સામેલ કરો
        - Answer માં ચોક્કસ આંકડાકીય મૂલ્યો આપો, પ્રશ્ન પુનરાવર્તન નહીં
        
        પાનાનું ટેક્સ્ટ:
        {page_text}
        
        JSON ફોર્મેટ:
        [
        {{
            "example_number": "ઉદાહરણ 19",
            "question": "એક હોડી નદીના સામા પ્રવાહે 30 કિમી અને પ્રવાહની દિશામાં 44 કિમી અંતર 10 કલાકમાં કાપે છે...",
            "answer": "હોડીની સ્થિર પાણીમાં ઝડપ = 8 કિમી/કલાક, નદીના પ્રવાહની ઝડપ = 3 કિમી/કલાક",
            "explanation": "ધારો કે હોડીની સ્થિર પાણીમાં ઝડપ x કિમી/કલાક અને પ્રવાહની ઝડપ y કિમી/કલાક છે. સમીકરણો: 30/(x-y) + 44/(x+y) = 10...",
            "question_type": "Long Answer",
            "mentioned_visuals": [
            {{
                "type": "કોષ્ટક/આકૃતિ/ચિત્ર",
                "reference": "કોષ્ટક 3.1",
                "context": "શા માટે જરૂરી છે"
            }}
            ]
        }}
        ]
        
        જો કોઈ ઉદાહરણ ન હોય તો [] આપો.
        """ 
        
        try:
            response = self.gemini_model.generate_content(prompt)
            self.api_calls["gemini_api"] += 1
            
            response_text = response.text.strip()
            
            if not response_text:
                return []
            
            # Clean markdown formatting
            if response_text.startswith("```"):
                response_text = response_text.strip("`")
                if response_text.lower().startswith("json"):
                    response_text = response_text[4:].strip()
            
            response_text = response_text.strip()
            if not (response_text.startswith('[') or response_text.startswith('{')):
                return []
            
            examples = json.loads(response_text)
            
            if not isinstance(examples, list):
                return []
            
            # Add metadata
            for example in examples:
                if isinstance(example, dict):
                    example["page_number"] = page_number
                    example["chapter"] = chapter_name
                    example["extracted_at"] = datetime.now().isoformat()
                    example["source"] = "example"
                    example["status"] = "inactive"
            
            return examples
            
        except json.JSONDecodeError:
            print(f"  JSON parsing error on page {page_number}")
            return []
        except Exception as e:
            print(f"  AI extraction error on page {page_number}: {str(e)[:50]}...")
            return []



    def _enhance_examples_with_detected_images(self, examples, page_data):
        """Cross-reference examples with detected images and add their descriptions"""
        
        page_images = page_data.get('images', [])
        if not page_images:
            return examples
        
        for example in examples:
            example_full_text = f"{example.get('question', '')} {example.get('explanation', '')}"
            
            # Find relevant images based on content matching
            detected_visuals = []
            
            for image in page_images:
                image_desc = image.get('educational_description', '')
                object_type = image.get('object_type', {})
                
                # Extract object description
                if isinstance(object_type, dict):
                    obj_description = object_type.get('description', '')
                else:
                    obj_description = str(object_type)
                
                # Check for relevance using keywords and context
                relevance_score = self._calculate_image_relevance(example_full_text, image_desc, obj_description)
                
                if relevance_score > 0.3:  # Threshold for relevance
                    visual_info = {
                        "type": self._classify_visual_type(image_desc),
                        "reference_id": image.get('reference_id', f"ચિત્ર_{example.get('page_number')}" ),
                        "description": image_desc,
                        "detection_method": image.get('detection_method', 'unknown'),
                        "relevance_score": round(relevance_score, 2)
                    }
                    detected_visuals.append(visual_info)
            
            # Add detected visuals to example
            example['detected_visuals'] = detected_visuals
            
            # Combine mentioned and detected visuals
            total_visuals = len(example.get('mentioned_visuals', [])) + len(detected_visuals)
            example['total_visual_references'] = total_visuals
        
        return examples

    def _calculate_image_relevance(self, example_text, image_desc, obj_description):
        """Calculate relevance score between example and image"""
        
        # Keywords that indicate mathematical content
        math_keywords = [
            'સમીકરણ', 'કોષ્ટક', 'આકૃતિ', 'આલેખ', 'ગ્રાફ', 'રેખા', 'બિંદુ',
            'ઉકેલ', 'હલ', 'સંખ્યા', 'મૂલ્ય', 'છેદ', 'intersection', 'coordinate'
        ]
        
        score = 0
        example_lower = example_text.lower()
        image_desc_lower = image_desc.lower()
        
        # Check for direct keyword matches
        for keyword in math_keywords:
            if keyword in example_lower and keyword in image_desc_lower:
                score += 0.2
        
        # Check for specific references (કોષ્ટક 3.1, આકૃતિ 3.2, etc.)
        import re
        ref_pattern = r'(કોષ્ટક|આકૃતિ|ચિત્ર|આલેખ)\s*\d+\.\d+'
        
        example_refs = re.findall(ref_pattern, example_text)
        image_refs = re.findall(ref_pattern, image_desc)
        
        common_refs = set(example_refs) & set(image_refs)
        score += len(common_refs) * 0.4
        
        # Check for equation mentions
        equation_pattern = r'\d*[xy]\s*[+\-]\s*\d*[xy]\s*=\s*\d+'
        if re.search(equation_pattern, example_text) and re.search(equation_pattern, image_desc):
            score += 0.3
        
        return min(score, 1.0)  # Cap at 1.0

    def _classify_visual_type(self, image_description):
        """Classify visual type based on description"""
        
        desc_lower = image_description.lower()
        
        if 'કોષ્ટક' in desc_lower or 'table' in desc_lower:
            return 'કોષ્ટક'
        elif 'આકૃતિ' in desc_lower or 'આલેખ' in desc_lower or 'graph' in desc_lower:
            return 'આકૃતિ/આલેખ' 
        elif 'ચિત્ર' in desc_lower or 'diagram' in desc_lower:
            return 'ચિત્ર'
        elif 'રેખા' in desc_lower or 'line' in desc_lower:
            return 'રેખાકૃતિ'
        else:
            return 'આકૃતિ'


    def _get_chapter_name(self, json_data):
        """Extract chapter name from JSON metadata"""
        # Try to get from chapter_info first
        chapter_summary = json_data.get('chapter_info', {}).get('chapter_summary', '')
        
        # Look for chapter name patterns in summary
        if 'દ્વિચલ સુરેખ સમીકરણ' in chapter_summary:
            return 'દ્વિચલ સુરેખ સમીકરણયુગ્મ'
        
        # Fallback: extract from filename
        source_pdf = json_data.get('metadata', {}).get('source_pdf', '')
        return source_pdf.replace('.pdf', '').replace('-', ' ').title()

    def save_examples(self, examples, output_file):
        """Save extracted examples to JSON file"""
        print(f"\n💾 Saving {len(examples)} examples to: {output_file}")
        
        output_data = {
            "metadata": {
                "extraction_type": "examples",
                "total_examples": len(examples),
                "extracted_at": datetime.now().isoformat(),
                "api_calls": self.api_calls["gemini_api"]
            },
            "examples": examples
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Statistics
        question_types = {}
        for example in examples:
            qtype = example.get('question_type', 'Unknown')
            question_types[qtype] = question_types.get(qtype, 0) + 1
        
        print(f"✅ EXAMPLES SAVED SUCCESSFULLY")
        print(f"📊 Question Type Distribution:")
        for qtype, count in question_types.items():
            print(f"  📝 {qtype}: {count}")

def main():
    """Main execution function"""
    print("📚 GSEB Example Extractor")
    print("🔍 Extract examples from processed textbook JSON")
    print("="*50)
    
    # Initialize extractor
    try:
        extractor = GSEBExampleExtractor()
    except ValueError as e:
        print(f"❌ Initialization Error: {str(e)}")
        return
    
    # Get input JSON file
    json_file = input("\n📂 Enter path to processed JSON file: ").strip()
    if not os.path.exists(json_file):
        print("❌ File not found!")
        return
    
    # Load and process
    json_data = extractor.load_processed_json(json_file)
    examples = extractor.extract_examples_from_chapter(json_data)    
    
    if examples:
        # Save examples
        output_file = f"extracted_examples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        extractor.save_examples(examples, output_file)
        print(f"\n✅ SUCCESS! Examples saved to: {output_file}")
    else:
        print("\n⚠️ No examples found in the document")

if __name__ == "__main__":
    main()