# exercise_extractor.py
import os
import json
import time
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

class GSEBExerciseExtractor:
    """
    Extract exercise questions from GSEB Mathematics textbook pages.
    Works for all mathematics chapters - algebra, geometry, trigonometry, statistics, etc.
    
    Features:
    - Detects exercise sections (સ્વાધ્યાય, અભ્યાસ, પ્રશ્નો)
    - Splits sub-questions into individual questions
    - Classifies into 9 question types using AI
    - Extracts visual references (આકૃતિ, આલેખ, કોષ્ટક)
    - Generates answers and explanations
    """
    
    def __init__(self):
        """Initialize the exercise extractor with Gemini API setup"""
        # Load environment variables
        load_dotenv()
        
        # Initialize Gemini AI
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Track API usage for cost monitoring
        self.api_calls = {"gemini_api": 0}
        
        # Define question types for validation
        self.valid_question_types = [
            "Very Short / Objective (O)",
            "MCQs (Multiple Choice Questions)",
            "True / False",
            "Fill in the Blanks", 
            "Short Answer – I (SA-I)",
            "Short Answer – II (SA-II)",
            "Long Answer (LA)",
            "Match the Values / (Jodka Jodo)",
            "Diagram-Based"
        ]
        
        print("📚 Initialized GSEB Exercise Extractor")
        print(f"🔑 Gemini API Key: {'✅ Loaded' if gemini_api_key else '❌ Missing'}")
        print(f"🎯 Target Subject: Mathematics (All Chapters)")

    def load_processed_json(self, json_file_path):
        """
        Load processed JSON from main.py output
        
        Args:
            json_file_path (str): Path to the processed chapter JSON file
            
        Returns:
            dict: Loaded JSON data with pages and metadata
        """
        print(f"\n📂 Loading processed JSON: {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 Loaded data:")
        print(f"  📄 Total pages: {len(data.get('pages', []))}")
        print(f"  📚 Chapter: {data.get('metadata', {}).get('source_pdf', 'Unknown')}")
        
        return data

    def extract_exercises_from_chapter(self, json_data):
        """
        Main method to extract exercises from all pages of a mathematics chapter
        
        Args:
            json_data (dict): Processed chapter data with page-by-page text
            
        Returns:
            list: List of extracted exercise questions with metadata
        """
        print("\n" + "="*50)
        print("🔍 EXERCISE EXTRACTION FROM MATHEMATICS CHAPTER")
        print("="*50)
        
        pages_data = json_data.get('pages', [])
        chapter_name = self._get_chapter_name(json_data)
        
        all_exercises = []
        
        # Process each page looking for exercise sections
        for page in tqdm(pages_data, desc="🔍 Processing pages for exercises"):
            try:
                page_number = page.get('page_number', 0)
                page_text = page.get('text', '')
                
                # Skip pages with insufficient content
                if len(page_text) < 100:
                    print(f"    ⏭️  Page {page_number}: Insufficient content, skipping")
                    continue
                    
                # # Check for exercise indicators in the page
                # if not self._has_exercise_content(page_text):
                #     continue
                
                print(f"    🎯 Page {page_number}: Exercise content detected")
                
                # Step 1: Extract exercises using AI
                exercises = self._extract_exercises_with_ai(page_text, chapter_name, page_number)
                
                if exercises:
                    # Step 2: Enhance with visual descriptions from page images
                    exercises = self._enhance_exercises_with_visual_content(exercises, page)
                    
                    # Log extraction results
                    for exercise in exercises:
                        qtype = exercise.get('question_type', 'Unknown')
                        visuals_count = len(exercise.get('mentioned_visuals', []))
                        print(f"      📝 Q{exercise.get('original_question_number', '')}"
                              f"({exercise.get('sub_question_number', '')}): {qtype}, "
                              f"{visuals_count} visuals")
                    
                    all_exercises.extend(exercises)
                
                # Rate limiting to avoid API overload
                time.sleep(1)
                
            except Exception as e:
                print(f"  ❌ Error processing page {page_number}: {str(e)[:100]}...")
                continue
        
        print(f"\n📚 EXTRACTION COMPLETED")
        print(f"📝 Total questions extracted: {len(all_exercises)}")
        
        # Print statistics
        # self._print_extraction_statistics(all_exercises)
        
        return all_exercises



    def _has_exercise_content(self, page_text):
        """
        Check if page contains actual exercise/practice sections (not just any questions)
        
        Args:
            page_text (str): Text content of the page
            
        Returns:
            bool: True if page has actual exercise section headers
        """
        import re
        
        # Strict exercise section patterns with numbers
        exercise_patterns = [
            r'સ્વાધ્યાય\s+\d+\.\d+',      # સ્વાધ્યાય 3.1, સ્વાધ્યાય 2.5
            r'અભ્યાસ\s+\d+\.\d+',         # અભ્યાસ 3.1
            r'પ્રેક્ટિસ\s+\d+\.\d+',       # પ્રેક્ટિસ 3.1
            r'Exercise\s+\d+\.\d+',        # Exercise 3.1
            r'કસોટી\s+\d+\.\d+'           # કસોટી 3.1
        ]
        
        # Check for specific exercise section headers
        for pattern in exercise_patterns:
            match = re.search(pattern, page_text)
            if match:
                found_section = match.group()
                print(f"    ✅ Found exercise section: {found_section}")
                return True
        
        # Secondary check for exercise indicators without numbers (less strict)
        basic_indicators = ['સ્વાધ્યાય', 'અભ્યાસ', 'પ્રશ્નો']
        
        for indicator in basic_indicators:
            if indicator in page_text:
                # Additional validation: check if followed by numbered questions
                if self._has_numbered_questions_after_indicator(page_text, indicator):
                    print(f"    ✅ Found exercise content with {indicator}")
                    return True
        
        print(f"    ⏭️  No exercise section found on this page")
        return False

    def _has_numbered_questions_after_indicator(self, page_text, indicator):
        """
        Check if there are numbered questions after the exercise indicator
        
        Args:
            page_text (str): Text content of the page
            indicator (str): Exercise indicator like 'સ્વાધ્યાય'
            
        Returns:
            bool: True if numbered questions found after indicator
        """
        import re
        
        # Find the position of the indicator
        indicator_pos = page_text.find(indicator)
        if indicator_pos == -1:
            return False
        
        # Look for numbered questions after the indicator
        text_after_indicator = page_text[indicator_pos:]
        
        # Patterns for numbered questions
        question_patterns = [
            r'\n\s*\d+\.\s+',          # 1. 2. 3.
            r'\n\s*\(\s*[ivx]+\s*\)',  # (i) (ii) (iii)
            r'\n\s*[અઆઇ]\)\s+'        # અ) આ) ઇ)
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, text_after_indicator[:500]):  # Check next 500 chars
                return True
        
        return False





    def _extract_exercises_with_ai(self, page_text, chapter_name, page_number):
        """
        Use Gemini AI to extract and classify exercise questions from page text
        
        Args:
            page_text (str): Text content of the page
            chapter_name (str): Name of the mathematics chapter
            page_number (int): Page number for reference
            
        Returns:
            list: List of extracted exercise questions with full metadata
        """
        print(f"    🤖 Using AI to extract exercises from page {page_number}")
        

        prompt = f"""
        તમે ધોરણ 10 ગુજરાતી માધ્યમના ગણિત પાઠ્યપુસ્તકમાંથી **ફક્ત સ્વાધ્યાય/અભ્યાસ વિભાગના** પ્રશ્નો કાઢી રહ્યા છો.

        **JSON આઉટપુટ ફરજિયાત નિયમો:**
        - તમારો જવાબ માત્ર JSON format માં હોવો જોઈએ
        - Explanation field માં line breaks (\\n) ન વાપરો
        - Mathematical equations અને symbols વાપરી શકો છો (x=3, y=2, +, -, =, etc.)
        - ફક્ત new lines અને tabs ટાળો

        **મહત્વપૂર્ણ નિયમો:**
        1. ફક્ત "સ્વાધ્યાય X.Y" અથવા "અભ્યાસ X.Y" હેડિંગ પછીના પ્રશ્નો જ કાઢો
        2. દરેક પ્રશ્ન માટે **ફરજિયાત** સંપૂર્ણ જવાબ અને સ્પષ્ટીકરણ આપો
        3. Exercise number ચોક્કસ શોધીને બહાર કાઢો
        4. **ઉપપ્રશ્ન બનાવતી વખતે મુખ્ય પ્રશ્નનો સંદર્ભ જોડો**

        **ઉપપ્રશ્ન નિયમો - અતિ મહત્વપૂર્ણ:**
        - મુખ્ય પ્રશ્ન: "નીચેના સમીકરણયુગ્મ હલ કરો:"
        - ઉપપ્રશ્ન (i): "x + y = 5, x - y = 1" 
        - **સંપૂર્ણ પ્રશ્ન બનાવો**: "નીચેના સમીકરણયુગ્મ હલ કરો: x + y = 5, x - y = 1"
        - **માત્ર equations જ ન લખો** - હંમેશા મુખ્ય instruction સાથે જોડો

        **Answer ફીલ્ડ:**
        - માત્ર અંતિમ પરિણામ લખો (જેમ કે: "x = 3, y = 2")

        **Explanation ફીલ્ડ:**
        - પગલાવાર ગાણિતિક ઉકેલ એક continuous text માં લખો
        - Mathematical equations સાચવી રાખો
        - પગલાઓ વચ્ચે "પછી" અથવા "અને" વાપરો

        **9 પ્રશ્ન પ્રકારો:**
        1. "Very Short / Objective (O)" - 1 ગુણ
        2. "MCQs (Multiple Choice Questions)" 
        3. "True / False" 
        4. "Fill in the Blanks" 
        5. "Short Answer – I (SA-I)" - 2 ગુણ
        6. "Short Answer – II (SA-II)" - 3 ગુણ
        7. "Long Answer (LA)" - 4+ ગુણ
        8. "Match the Values / (Jodka Jodo)" 
        9. "Diagram-Based"

        પાનાનું લખાણ:
        {page_text}

        **ફક્ત નીચેના JSON ફોર્મેટ માં આપો:**
        [{{"exercise_number": "3.1", "page_number": {page_number}, "original_question_number": "1", "sub_question_number": "i", "question_text": "નીચેના સમીકરણયુગ્મ હલ કરો: x + y = 5 અને x - y = 1", "question_type": "Short Answer – II (SA-II)", "answer": "x = 3, y = 2", "explanation": "આપેલ x + y = 5 અને x - y = 1, બંને સમીકરણ ઉમેરતાં 2x = 6 તેથી x = 3, પછી x = 3 પ્રથમ સમીકરણમાં મૂકતાં 3 + y = 5 તેથી y = 2, આ ઉમેરવાની પદ્ધતિ છે.", "marks_estimate": 3, "difficulty": "Medium", "mentioned_visuals": []}}]

        **ચેતવણી:**
        - Mathematical equations જાળવી રાખો
        - ફક્ત line breaks ટાળો, equations નહીં
        - જો કોઈ સ્વાધ્યાય વિભાગ ન હોય તો માત્ર [] આપો
        - **ઉપપ્રશ્નો માં હંમેશા મુખ્ય instruction શામેલ કરો**
        """



        try:
            response = self.gemini_model.generate_content(prompt)
            self.api_calls["gemini_api"] += 1
            
            response_text = response.text.strip()
            
            if not response_text:
                return []
            
            # Clean markdown formatting if present
            response_text = self._clean_response_text(response_text)
            
        # Parse JSON response
            exercises = json.loads(response_text)
            
            if not isinstance(exercises, list):
                return []
            
            # Filter and validate exercises before adding metadata
            validated_exercises = []
            for exercise in exercises:
                if isinstance(exercise, dict):
                    exercise["chapter"] = chapter_name
                    exercise["extracted_at"] = datetime.now().isoformat()
                    exercise["source"] = "exercise"
                    exercise["status"] = "inactive"
                    validated_exercises.append(exercise)
                else:
                    print(f"      ⚠️  Skipping invalid exercise type: {type(exercise)}")
            
            print(f"      ✅ Successfully extracted {len(validated_exercises)} exercises")
            return validated_exercises

            
        except json.JSONDecodeError as e:
            print(f"      ❌ JSON parsing error: {str(e)}")
            return []
        except Exception as e:
            print(f"      ❌ AI extraction error: {str(e)}")
            return []


    def _clean_response_text(self, response_text):
        """
        Clean markdown formatting from Gemini response
        
        Args:
            response_text (str): Raw response from Gemini
            
        Returns:
            str: Cleaned JSON text
        """
        # Remove markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.strip("`")
            if response_text.lower().startswith("json"):
                response_text = response_text[4:].strip()
        
        return response_text.strip()


    def _enhance_exercises_with_visual_content(self, exercises, page_data):
        """
        Enhance exercises with actual visual descriptions from page images
        
        Args:
            exercises (list): List of extracted exercises
            page_data (dict): Page data containing images
            
        Returns:
            list: Enhanced exercises with visual descriptions
        """
        page_images = page_data.get('images', [])
        if not page_images:
            return exercises
        
        print(f"      🖼️  Enhancing {len(exercises)} exercises with visual content")
        
        enhanced_exercises = []
        
        for exercise in exercises:
            # Add type checking to handle both dict and string cases
            if not isinstance(exercise, dict):
                print(f"        ⚠️  Skipping invalid exercise type: {type(exercise)}")
                continue
            
            mentioned_visuals = exercise.get('mentioned_visuals', [])
            
            # Ensure mentioned_visuals is a list
            if not isinstance(mentioned_visuals, list):
                mentioned_visuals = []
                exercise['mentioned_visuals'] = mentioned_visuals
            
            for visual in mentioned_visuals:
                # Ensure visual is a dictionary
                if not isinstance(visual, dict):
                    continue
                    
                visual_ref = visual.get('reference', '')
                visual_type = visual.get('type', '')
                
                # Find matching description from page images
                matching_desc = self._find_matching_visual_description(
                    visual_ref, visual_type, page_images
                )
                
                if matching_desc:
                    visual['full_description'] = matching_desc
                    print(f"        ✅ Found description for {visual_ref}")
                else:
                    visual['full_description'] = "વર્ણન ઉપલબ્ધ નથી"
                    print(f"        ⚠️  No description found for {visual_ref}")
            
            enhanced_exercises.append(exercise)
        
        return enhanced_exercises




    def _find_matching_visual_description(self, visual_reference, visual_type, page_images):
        """
        Find matching image description based on reference and type
        
        Args:
            visual_reference (str): Reference like "આકૃતિ 3.5"
            visual_type (str): Type like "આકૃતિ", "આલેખ"
            page_images (list): List of detected images on the page
            
        Returns:
            str or None: Matching description if found
        """
        import re
        
        # Method 1: Exact reference matching
        if visual_reference:
            ref_match = re.search(r'\d+\.\d+', visual_reference)
            if ref_match:
                ref_number = ref_match.group()
                ref_type = visual_reference.split()[0]  # આકૃતિ, આલેખ
                
                for image in page_images:
                    img_desc = image.get('educational_description', '')
                    if ref_number in img_desc and ref_type in img_desc:
                        return img_desc
        
        # Method 2: Type-based matching
        visual_type_lower = visual_type.lower()
        
        type_keywords = {
            'આકૃતિ': ['આકૃતિ', 'figure', 'diagram'],
            'આલેખ': ['આલેખ', 'graph', 'chart'],
            'કોષ્ટક': ['કોષ્ટક', 'table']
        }
        
        if visual_type_lower in type_keywords:
            keywords = type_keywords[visual_type_lower]
            
            for image in page_images:
                img_desc = image.get('educational_description', '').lower()
                if any(keyword in img_desc for keyword in keywords):
                    return image.get('educational_description', '')
        
        return None

    def _get_chapter_name(self, json_data):
        """
        Extract chapter name from JSON metadata
        
        Args:
            json_data (dict): Processed chapter JSON data
            
        Returns:
            str: Chapter name for context
        """
        # Try to get from chapter_info first
        chapter_summary = json_data.get('chapter_info', {}).get('chapter_summary', '')
        
        if chapter_summary:
            return chapter_summary.split('.')[0] if '.' in chapter_summary else chapter_summary[:50]
        
        # Fallback: extract from filename
        source_pdf = json_data.get('metadata', {}).get('source_pdf', '')
        return source_pdf.replace('.pdf', '').replace('-', ' ').title()

    def _print_extraction_statistics(self, exercises):
        """
        Print detailed statistics about extracted exercises
        
        Args:
            exercises (list): List of extracted exercises
        """
        if not exercises:
            return
            
        # Question type distribution
        question_types = {}
        difficulties = {}
        total_visuals = 0
        
        for exercise in exercises:
            # Count question types
            qtype = exercise.get('question_type', 'Unknown')
            question_types[qtype] = question_types.get(qtype, 0) + 1
            
            # Count difficulties
            difficulty = exercise.get('difficulty', 'Unknown')
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
            
            # Count visual references
            total_visuals += len(exercise.get('mentioned_visuals', []))
        
        print(f"\n📊 EXTRACTION STATISTICS:")
        print(f"  📝 Total Questions: {len(exercises)}")
        print(f"  🖼️  Total Visual References: {total_visuals}")
        print(f"  🤖 AI API Calls: {self.api_calls['gemini_api']}")
        
        print(f"\n📋 Question Type Distribution:")
        for qtype, count in sorted(question_types.items()):
            print(f"  📄 {qtype}: {count}")
        
        print(f"\n🎯 Difficulty Distribution:")
        for diff, count in sorted(difficulties.items()):
            print(f"  🔘 {diff}: {count}")

    def save_exercises(self, exercises, output_file):
        """
        Save extracted exercises to JSON file with metadata
        
        Args:
            exercises (list): List of extracted exercises
            output_file (str): Output file path
        """
        print(f"\n💾 Saving {len(exercises)} exercises to: {output_file}")
        
        # Create output data structure
        output_data = {
            "metadata": {
                "extraction_type": "exercises",
                "subject": "Mathematics",
                "total_questions": len(exercises),
                "extracted_at": datetime.now().isoformat(),
                "api_calls": self.api_calls["gemini_api"],
                "extractor_version": "1.0"
            },
            "exercises": exercises
        }
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ EXERCISES SAVED SUCCESSFULLY")
        self._print_save_summary(exercises)

    def _print_save_summary(self, exercises):
        """
        Print summary of saved exercises
        
        Args:
            exercises (list): List of saved exercises
        """
        # Quick statistics for saved data
        exercise_numbers = set()
        unique_pages = set()
        
        for exercise in exercises:
            if exercise.get('exercise_number'):
                exercise_numbers.add(exercise['exercise_number'])
            if exercise.get('page_number'):
                unique_pages.add(exercise['page_number'])
        
        print(f"📈 Save Summary:")
        print(f"  📚 Exercise Sets: {len(exercise_numbers)}")
        print(f"  📄 Pages Processed: {len(unique_pages)}")
        print(f"  💰 Total API Cost: ~${self.api_calls['gemini_api'] * 0.01:.2f}")

def main():
    """Main execution function for the exercise extractor"""
    print("📚 GSEB Mathematics Exercise Extractor")
    print("🔍 Extract exercises from processed mathematics textbook JSON")
    print("🎯 Supports all mathematics chapters and topics")
    print("="*60)
    
    # Initialize extractor
    try:
        extractor = GSEBExerciseExtractor()
    except ValueError as e:
        print(f"❌ Initialization Error: {str(e)}")
        return
    
    # Get input JSON file from user
    json_file = input("\n📂 Enter path to processed JSON file: ").strip()
    if not os.path.exists(json_file):
        print("❌ File not found!")
        return
    
    print(f"✅ File found: {json_file}")
    
    # Load and process the chapter data
    try:
        print("\n🔄 Loading chapter data...")
        json_data = extractor.load_processed_json(json_file)
        
        print("🔄 Starting exercise extraction...")
        exercises = extractor.extract_exercises_from_chapter(json_data)    
        
        if exercises:
            # Generate output filename with timestamp
            output_file = f"extracted_exercises_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            extractor.save_exercises(exercises, output_file)
            print(f"\n🎉 SUCCESS! Exercises saved to: {output_file}")
        else:
            print("\n⚠️  No exercises found in the document")
            print("💡 Make sure the document contains સ્વાધ્યાય/અભ્યાસ sections")
            
    except Exception as e:
        print(f"❌ Processing Error: {str(e)}")
        print("💡 Please check the JSON file format and try again")

if __name__ == "__main__":
    main()