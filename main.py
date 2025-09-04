import os
import json
import time
import requests
import base64
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
from pypdf import PdfReader, PdfWriter  # For PDF page count
from pdf2image import convert_from_path  # For converting PDF to images
from io import BytesIO

# Load environment variables
load_dotenv()

class GSEBPDFProcessor:
    def __init__(self):
        # Initialize Google Cloud Vision API key
        self.vision_api_key = os.getenv('GOOGLE_CLOUD_VISION_API_KEY')
        if not self.vision_api_key:
            raise ValueError("GOOGLE_CLOUD_VISION_API_KEY not found in environment variables")
        
        # Initialize Gemini
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # API call counters
        self.api_calls = {
            "vision_api": 0,
            "gemini_api": 0,
            "total_tokens_used": 0,
            "start_time": datetime.now()
        }
        
        # Class 10 Maths Gujarati specific prompts
        self.prompts = self._load_subject_prompts()
        
        print("🔧 Initialized GSEB PDF Processor")
        print(f"🔑 Vision API Key: {'✅ Loaded' if self.vision_api_key else '❌ Missing'}")
        print(f"🔑 Gemini API Key: {'✅ Loaded' if gemini_api_key else '❌ Missing'}")
        print("📊 API Call Counter: Initialized")
        
    def _load_subject_prompts(self):
        """Load Class 10 Mathematics Gujarati specific prompts"""
        return {
            "page_summarization": """
            તમે વર્ગ 10ના ગણિત વિષયનું પુસ્તક (ગુજરાતી માધ્યમ) નું વિશ્લેષણ કરો છો.
            આ પાનાની સામગ્રીનો સારાંશ ગુજરાતીમાં આપો, જેમાં નીચેની બાબતો સમાવેશ કરો:
            
            - મુખ્ય ગણિતીય સંકલ્પનાઓ (Key mathematical concepts)
            - સૂત્રો અને નિયમો (Formulas and rules)
            - ઉદાહરણો (Examples) - જો કોઈ હોય તો
            - કસોટીઓ (Exercises) - જો કોઈ હોય તો  
            - ચિત્રો/આકૃતિઓ (Diagrams) - જો કોઈ હોય તો
            
            પાનાની સામગ્રી: {page_text}
            
            ચિત્રોની માહિતી: {image_descriptions}
            
            ગુજરાતીમાં સારાંશ આપો:
            """,
            "chapter_analysis": """
            તમે વર્ગ 10ના ગણિત અધ્યાયનું વિશ્લેષણ કરો છો. તમામ પાનાઓના સારાંશના આધારે નીચેની માહિતી આપો:
            
            પાનાઓના સારાંશ: {page_summaries}
            
            કૃપા કરીને આપો:
            
            **અધ્યાયનો સંપૂર્ણ સારાંશ:**
            [સંપૂર્ણ અધ્યાયનો સારાંશ]
            
            **મુખ્ય વિષયોની યાદી:**
            1. [વિષય 1]
            2. [વિષય 2]
            3. [વિષય 3]
            
            **શીખવાના પરિણામો:**
            - [પરિણામ 1]
            - [પરિણામ 2]
            """,
            "topic_assignment": """
            આ પાનાની સામગ્રી અને વિષયોની યાદીના આધારે, આ પાનાને સંબંધિત વિષયોના નંબર આપો.
            
            નિયમો:
            - એક પાનામાં બહુવિધ વિષયો હોઈ શકે
            - એક જ વિષય બે વખત ન આવવો જોઈએ
            - ફક્ત સૌથી સંબંધિત વિષયો પસંદ કરો
            
            પાનાની સામગ્રી: {page_text}
            
            ઉપલબ્ધ વિષયો:
            {topics_list}
            
            ફક્ત વિષય નંબરો આપો (comma separated):
            """,
            "image_description": """
            આ ગણિતના ચિત્રનું શૈક્ષણિક વર્ણન ગુજરાતીમાં આપો:
            
            ચિત્રમાં જે દેખાય છે: {image_content}
            
            સંદર્ભ: વર્ગ 10 ગણિત - {context}
            
            ગુજરાતીમાં શૈક્ષણિક વર્ણન આપો (શું દેખાય છે, કેવા ઉપયોગ માટે છે):
            """
        }
    

    def extract_pdf_with_images(self, pdf_path):
        """Extract text and images from PDF with enhanced detection capabilities"""
        print("\n" + "="*50)
        print("📄 STEP 1: ENHANCED PDF EXTRACTION WITH IMAGES (PAGE BY PAGE)")
        print("="*50)
        print(f"📄 Processing file: {os.path.basename(pdf_path)}")
        print(f"📊 API Call Status - Vision: {self.api_calls['vision_api']}, Gemini: {self.api_calls['gemini_api']}")
        
        start_time = time.time()
        
        # Convert PDF to images using pdf2image
        try:
            images = convert_from_path(pdf_path, dpi=300, fmt='png')
            print(f"📄 Converted {len(images)} pages to images")
        except Exception as e:
            print(f"❌ Failed to convert PDF to images: {str(e)[:100]}...")
            return []
        
        pages_data = []
        
        for page_idx, image in enumerate(tqdm(images, desc="Processing pages")):
            # Convert image to bytes
            img_buffer = BytesIO()
            image.save(img_buffer, format="PNG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            img_buffer.close()
            
            file_size = len(img_base64) / (1024 * 1024)  # MB
            print(f"📄 Page {page_idx+1} image size: {file_size:.2f} MB")
            
            # ENHANCED: Prepare API request with multiple detection features
            api_url = f"https://vision.googleapis.com/v1/images:annotate?key={self.vision_api_key}"
            
            request_payload = {
                "requests": [
                    {
                        "image": {
                            "content": img_base64
                        },
                        "features": [
                            {
                                "type": "TEXT_DETECTION",
                                "maxResults": 100
                            },
                            {
                                "type": "DOCUMENT_TEXT_DETECTION"  # NEW: Better for structured content
                            },
                            {
                                "type": "OBJECT_LOCALIZATION",
                                "maxResults": 20
                            }
                        ],
                        "imageContext": {
                            "languageHints": ["gu", "en"]  # Gujarati and English
                        }
                    }
                ]
            }
            
            print(f"🚀 Sending enhanced image request for page {page_idx+1} to Google Vision API...")
            headers = {'Content-Type': 'application/json'}
            response = requests.post(api_url, json=request_payload, headers=headers)
            
            # Count Vision API call
            self.api_calls["vision_api"] += 1
            print(f"📊 Vision API Call #{self.api_calls['vision_api']} completed for page {page_idx+1}")
            
            if response.status_code != 200:
                print(f"❌ Vision API Error for page {page_idx+1}: {response.status_code}")
                print(f"🔍 Error details: {response.text[:200]}...")
                pages_data.append({
                    "page_number": page_idx + 1,
                    "text": "",
                    "images": [],
                    "extracted_at": datetime.now().isoformat(),
                    "error": f"Vision API error: {response.status_code} - {response.text[:200]}"
                })
                continue
            
            result = response.json()
            print(f"🔍 API Response for page {page_idx+1}: {json.dumps(result, ensure_ascii=False, indent=2)[:500]}...")
            
            # ENHANCED: Process response with multiple detection methods
            page_data = self._process_enhanced_vision_response(result, page_idx + 1)
            pages_data.append(page_data)
            time.sleep(1)  # Rate limiting
        
        processing_time = time.time() - start_time
        total_images = sum(len(page.get('images', [])) for page in pages_data)
        total_characters = sum(len(page.get('text', '')) for page in pages_data)
        
        print("\n✅ ENHANCED PDF EXTRACTION COMPLETED")
        print(f"📄 Pages extracted: {len(pages_data)} / {len(images)}")
        print(f"🖼️ Images/diagrams detected: {total_images}")
        print(f"🔍 Total characters extracted: {total_characters}")
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"📊 Total Vision API calls so far: {self.api_calls['vision_api']}")
        
        return pages_data

    def _process_enhanced_vision_response(self, result, page_number):
        """Process enhanced Vision API response with multiple detection methods"""
        print(f"🔍 Processing enhanced Vision API response for page {page_number}")
        
        page_data = {
            "page_number": page_number,
            "text": "",
            "images": [],
            "extracted_at": datetime.now().isoformat()
        }
        
        if 'responses' not in result or len(result['responses']) == 0:
            print(f"⚠️ No response data for page {page_number}")
            page_data["error"] = "No response data from Vision API"
            return page_data
        
        page_response = result['responses'][0]
        
        # Method 1: Extract text using TEXT_DETECTION (existing)
        if 'textAnnotations' in page_response and page_response['textAnnotations']:
            page_data["text"] = page_response['textAnnotations'][0].get('description', '')
            print(f"📄 Page {page_number}: {len(page_data['text'])} characters extracted (TEXT_DETECTION)")
        else:
            print(f"⚠️ No textAnnotations for page {page_number}")
            page_data["error"] = "No text extracted from TEXT_DETECTION"
        
        # Method 2: Enhanced text extraction using DOCUMENT_TEXT_DETECTION
        if 'fullTextAnnotation' in page_response:
            document_text = page_response['fullTextAnnotation'].get('text', '')
            if len(document_text) > len(page_data["text"]):
                page_data["text"] = document_text
                print(f"📄 Page {page_number}: Enhanced text extraction - {len(document_text)} characters (DOCUMENT_TEXT_DETECTION)")
        
        # Method 3: Object detection (existing but enhanced)
        detected_objects = []
        if 'localizedObjectAnnotations' in page_response:
            for obj in page_response['localizedObjectAnnotations']:
                print(f"🖼️ Page {page_number}: Detected object '{obj['name']}' (confidence: {obj.get('score', 0):.2f})")
                # Enhanced object type detection
                if self._is_mathematical_content(obj['name']):
                    detected_objects.append({
                        "object_type": obj['name'],
                        "confidence": obj.get('score', 0),
                        "detection_method": "vision_object_detection",
                        "raw_detection": obj['name']
                    })
                    print(f"🖼️ Page {page_number}: Added mathematical object '{obj['name']}' to detection list")
        
        # Store initial detections (will be enhanced by Gemini later)
        page_data["images"] = detected_objects
        
        chapter_name = getattr(self, "current_chapter", "અજ્ઞાત અધ્યાય")  # You can set before processing
        ai_detected = self.detect_mathematical_content_with_ai(page_data["text"], chapter_name, page_number)

        for det in ai_detected:
            page_data["images"].append({
                "object_type": det,
                "confidence": 1.0,
                "detection_method": "gemini_ai_detection",
                "raw_detection": det
            })
        
        return page_data


    def _is_mathematical_content(self, object_name):
        """Check if detected object is likely mathematical content (English labels from Vision API)"""
        
        # English keywords (from Vision API object detection)
        english_mathematical_keywords = [
            'diagram', 'chart', 'graph', 'table', 'mathematical expression', 
            'formula', 'image', 'figure', 'equation', 'plot', 'grid',
            'coordinate', 'axis', 'line', 'curve', 'geometric', 'triangle',
            'circle', 'rectangle', 'polygon', 'shape', 'drawing', 'illustration'
        ]
        
        # Gujarati keywords (if Vision API occasionally returns Gujarati labels)
        gujarati_mathematical_keywords = [
            'આકૃતિ', 'ચાર્ટ', 'આલેખ', 'કોષ્ટક', 'સૂત્ર', 'સમીકરણ',
            'ત્રિકોણ', 'વર્તુળ', 'ચતુર્ભુજ', 'રેખાકૃતિ', 'ચિત્ર'
        ]
        
        object_lower = object_name.lower()
        
        # Check English keywords (primary check)
        for keyword in english_mathematical_keywords:
            if keyword in object_lower:
                print(f"✅ Mathematical content detected (English): '{object_name}' contains '{keyword}'")
                return True
        
        # Check Gujarati keywords (fallback check)
        for keyword in gujarati_mathematical_keywords:
            if keyword in object_name:
                print(f"✅ Mathematical content detected (Gujarati): '{object_name}' contains '{keyword}'")
                return True
        
        print(f"❌ Non-mathematical content: '{object_name}'")
        return False


        

    def detect_mathematical_content_with_ai(self, page_text, chapter_name, page_number):
        """Detect and describe mathematical diagrams in Gujarati with chapter-aware hints"""
        
        chapter_content_map = {
        "વાસ્તવિક સંખ્યાઓ": [
            "સંખ્યા રેખા પર દર્શાવેલી સંખ્યાઓ",
            "મૂળાંક સાથે સંબંધિત આકૃતિઓ"
        ],
        "બહુપદીઓ": [
            "બહુપદીઓના ગ્રાફ",
            "મૂળ અને ગુણાકાર દર્શાવતી આકૃતિઓ"
        ],
        "દ્વિચલ સુરેખ સમીકરણયુગ્મ": [
            "સમીકરણના ગ્રાફ",
            "કોષ્ટક",
            "રેખાઓના intersection"
        ],
        "દ્વિઘાત સમીકરણ": [
            "પેરાબોલાનો ગ્રાફ",
            "વિભિન્ન કેસ માટેના ગ્રાફ (વાસ્તવિક મૂળ, કલ્પિત મૂળ)",
            "વર્ટેક્સ દર્શાવતો ગ્રાફ"
        ],
        "સમાન્તર શ્રેણી": [
            "અનુક્રમ દર્શાવતી કોષ્ટક",
            "આનુક્રમિક સંખ્યાઓ દર્શાવતો આલેખ"
        ],
        "ત્રિકોણ": [
            "ત્રિકોણની રચના",
            "સમાનતા દર્શાવતી આકૃતિઓ",
            "પ્રમાણ દર્શાવતી આકૃતિઓ"
        ],
        "યામ ભૂમિતિ": [
            "અક્ષાંકો પરના બિંદુઓ",
            "અંતર સૂત્ર",
            "વિભાગ સૂત્ર",
            "ત્રિકોણનું ક્ષેત્રફળ દર્શાવતી આકૃતિ"
        ],
        "ત્રિકોણમિતિ નો પરિચય": [
            "સમકોણ ત્રિકોણમાં ત્રિકોણમિતીય ગુણોત્તર",
            "યૂનિટ સર્કલ"
        ],
        "ત્રિકોણમિતિ ના ઉપયોગો": [
            "ઊંચાઈ અને અંતર દર્શાવતી આકૃતિઓ",
            "કોણ ઉન્નતિ અને અવનીતિ"
        ],
        "વર્તુળ": [
            "વર્તુળ",
            "સ્પર્શક",
            "જ્યોતિ",
            "ત્રજ્યા"
        ],
        "રચના": [
            "કંપાસથી રચના",
            "કોણ દ્વિભાજક",
            "ત્રિકોણની રચના"
        ],
        "વર્તુળ સંબંધિત ક્ષેત્રફળ": [
            "વર્તુળનો ક્ષેત્રફળ",
            "સેક્ટર",
            "સેગમેન્ટ"
        ],
        "પૃષ્ઠફળ અને ઘનફળ": [
            "ઘનાકૃતિઓના આલેખ",
            "સિલિન્ડર",
            "શંકુ",
            "ગોળાકાર"
        ],
        "આંકડાશાસ્ત્ર": [
            "હિસ્ટોગ્રામ",
            "બાર ચાર્ટ",
            "આવર્તન કોષ્ટક",
            "પાઈ ચાર્ટ"
        ],
        "સંભાવના": [
            "સંભાવના વૃક્ષ",
            "પ્રયોગોના આકૃતિઓ",
            "નમૂના સ્થાન"
            ]
        }
        
        expected_content = chapter_content_map.get(chapter_name, [])
        
        prompt = f"""
        તમે ધોરણ 10 ગણિતના પાઠ્યપુસ્તકના અધ્યાય "{chapter_name}" નું પાનું {page_number} વાંચી રહ્યા છો. 

        આ અધ્યાયમાં સામાન્ય રીતે જોવા મળતી આકૃતિઓ: {', '.join(expected_content)}.

        નીચેના લખાણને આધારે ઓળખો કે આ પાનામાં કઈ આકૃતિઓ છે. 
        દરેક માટે સ્પષ્ટ ગુજરાતી વર્ણન અને શૈક્ષણિક હેતુ આપો.

        લખાણ:
        {page_text}

        ફક્ત આ ફોર્મેટમાં JSON return કરો (કોઈ વધારાનો લખાણ નહીં):
        [
        {{
            "description": "x + y = 5 નું સુરેખ સમીકરણ દર્શાવતો ગ્રાફ જેમાં રેખા (0,5) અને (5,0) પરથી પસાર થાય છે.",
            "educational_context": "વિદ્યાર્થીઓ સમીકરણને ગ્રાફિક રીતે કેવી રીતે રજૂ થાય છે તે સમજી શકે છે."
        }}
        ]
        જો આકૃતિ ન હોય તો ખાલી array [] return કરો.
        """
        
        response = self.gemini_model.generate_content(prompt)
        text = response.text.strip()
        
        # 🛠️ Clean out markdown fences if Gemini adds them
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        
        try:
            detections = json.loads(text)
            return detections if isinstance(detections, list) else []
        except Exception:
            print(f"⚠️ Gemini response parsing error (cleaned): {text[:200]}")
            return []

  
    def describe_images_with_ai(self, pages_data):
        """Generate educational descriptions for detected images using Gemini API"""
        print("\n" + "="*50)
        print("🖼️ STEP 2: IMAGE DESCRIPTION GENERATION")
        print("="*50)
        
        total_images = sum(len(page.get('images', [])) for page in pages_data)
        print(f"🖼️ Total images to describe: {total_images}")
        print(f"📊 Current API calls - Vision: {self.api_calls['vision_api']}, Gemini: {self.api_calls['gemini_api']}")
        
        if total_images == 0:
            print("ℹ️ No images detected. Skipping image description step.")
            return pages_data
        
        start_time = time.time()
        successful_descriptions = 0
        failed_descriptions = 0
        
        for page in tqdm(pages_data, desc="Describing images"):
            if not page.get("images"):
                print(f"📄 Page {page['page_number']}: No images to describe")
                continue
            
            print(f"📄 Page {page['page_number']}: Processing {len(page['images'])} images")
            for image in page["images"]:
                try:
                    object_type = image["object_type"]
                    confidence = image["confidence"]
                    context = f"Page {page['page_number']} of Class 10 Mathematics textbook"
                    
                    prompt = self.prompts["image_description"].format(
                        image_content=f"{object_type} (confidence: {confidence:.2f})",
                        context=context
                    )
                    
                    print(f"  🔄 Sending image description request for {object_type} to Gemini API...")
                    response = self.gemini_model.generate_content(prompt)
                    
                    self.api_calls["gemini_api"] += 1
                    successful_descriptions += 1
                    
                    image["educational_description"] = response.text
                    print(f"  ✅ Description generated for {object_type} ({len(response.text)} characters)")
                    
                    time.sleep(1)
                except Exception as e:
                    failed_descriptions += 1
                    print(f"  ❌ Error describing {object_type}: {str(e)[:100]}...")
                    image["educational_description"] = "વર્ણન ઉપલબ્ધ નથી"
        
        processing_time = time.time() - start_time
        
        print("\n✅ IMAGE DESCRIPTION COMPLETED")
        print(f"🖼️ Successful descriptions: {successful_descriptions}")
        print(f"❌ Failed descriptions: {failed_descriptions}")
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"📊 Gemini API calls for descriptions: {successful_descriptions}")
        print(f"📊 Total Gemini API calls so far: {self.api_calls['gemini_api']}")
        
        return pages_data
    
    def integrate_images_in_text(self, pages_data):
        """Integrate image references into page text"""
        print("\n" + "="*50)
        print("🔗 STEP 3: TEXT-IMAGE INTEGRATION")
        print("="*50)
        
        total_pages_with_images = sum(1 for page in pages_data if page.get("images"))
        total_image_refs = sum(len(page.get("images", [])) for page in pages_data)
        
        print(f"📄 Pages with images: {total_pages_with_images}")
        print(f"🔗 Image references to add: {total_image_refs}")
        
        if total_image_refs == 0:
            print("ℹ️ No images to integrate. Skipping integration step.")
            return pages_data
        
        start_time = time.time()
        integrated_count = 0
        
        for page in pages_data:
            if page["images"]:
                print(f"📄 Page {page['page_number']}: Integrating {len(page['images'])} image references")
                
                image_refs = []
                for i, image in enumerate(page["images"], 1):
                    ref_text = f"\n[ચિત્ર {i}: {image['educational_description']}]"
                    image_refs.append(ref_text)
                    integrated_count += 1
                
                original_length = len(page["text"])
                page["text"] += "".join(image_refs)
                new_length = len(page["text"])
                
                print(f"  ✅ Added {len(image_refs)} references (+{new_length - original_length} characters)")
                
                for i, image in enumerate(page["images"], 1):
                    image["reference_id"] = f"ચિત્ર_{page['page_number']}_{i}"
        
        processing_time = time.time() - start_time
        
        print("\n✅ TEXT-IMAGE INTEGRATION COMPLETED")
        print(f"🔗 Total image references integrated: {integrated_count}")
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print("📊 No API calls required for this step")
        
        return pages_data
    
    def summarize_pages(self, pages_data):
        """Summarize each page using Gemini with image context"""
        print("\n" + "="*50)
        print("📝 STEP 4: PAGE-WISE SUMMARIZATION")
        print("="*50)
        
        print(f"📄 Total pages to summarize: {len(pages_data)}")
        print(f"📊 Current API calls - Vision: {self.api_calls['vision_api']}, Gemini: {self.api_calls['gemini_api']}")
        
        start_time = time.time()
        successful_summaries = 0
        failed_summaries = 0
        
        for page in tqdm(pages_data, desc="📝 Summarizing pages"):
            try:
                print(f"\n📄 Processing Page {page['page_number']}")
                
                image_descriptions = ""
                if page["images"]:
                    descriptions = [img["educational_description"] for img in page["images"]]
                    image_descriptions = "\n".join(descriptions)
                    print(f"  🖼️ Including {len(page['images'])} image descriptions")
                else:
                    image_descriptions = "આ પાનામાં કોઈ ચિત્ર નથી."
                    print("  📄 No images on this page")
                
                text_length = len(page["text"])
                print(f"  📊 Text length: {text_length} characters")
                
                prompt = self.prompts["page_summarization"].format(
                    page_text=page["text"],
                    image_descriptions=image_descriptions
                )
                
                print(f"  🔄 Sending to Gemini API...")
                response = self.gemini_model.generate_content(prompt)
                
                self.api_calls["gemini_api"] += 1
                successful_summaries += 1
                
                page["page_summary"] = response.text
                page["summarized_at"] = datetime.now().isoformat()
                
                summary_length = len(response.text)
                print(f"  ✅ Summary generated ({summary_length} characters) - API call #{self.api_calls['gemini_api']}")
                
                time.sleep(1)
            except Exception as e:
                failed_summaries += 1
                print(f"  ❌ Error: {str(e)[:100]}...")
                page["page_summary"] = "સારાંશ ઉપલબ્ધ નથી"
        
        processing_time = time.time() - start_time
        
        print("\n✅ PAGE SUMMARIZATION COMPLETED")
        print(f"📝 Successful summaries: {successful_summaries}")
        print(f"❌ Failed summaries: {failed_summaries}")
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"📊 Gemini API calls for summaries: {successful_summaries}")
        print(f"📊 Total Gemini API calls so far: {self.api_calls['gemini_api']}")
        
        return pages_data
    
    def analyze_chapter(self, pages_data):
        """Generate chapter analysis from all page summaries"""
        print("\n" + "="*50)
        print("📚 STEP 5: CHAPTER ANALYSIS")
        print("="*50)
        
        page_summaries = []
        valid_summaries = 0
        
        for page in pages_data:
            if 'page_summary' in page and page['page_summary'] != "સારાંશ ઉપલબ્ધ નથી":
                page_summaries.append(f"પાનું {page['page_number']}: {page['page_summary']}")
                valid_summaries += 1
        
        print(f"📄 Valid page summaries: {valid_summaries}/{len(pages_data)}")
        print(f"📊 Current API calls - Vision: {self.api_calls['vision_api']}, Gemini: {self.api_calls['gemini_api']}")
        
        if not page_summaries:
            print("❌ No valid page summaries found. Cannot analyze chapter.")
            return {"chapter_summary": "અધ્યાય વિશ્લેષણમાં ભૂલ - કોઈ સારાંશ ઉપલબ્ધ નથી", "analyzed_at": datetime.now().isoformat()}
        
        summaries_text = "\n\n".join(page_summaries)
        total_chars = len(summaries_text)
        print(f"📊 Total summary text: {total_chars} characters")
        
        start_time = time.time()
        
        try:
            print("🔄 Sending chapter analysis request to Gemini API...")
            
            prompt = self.prompts["chapter_analysis"].format(
                page_summaries=summaries_text
            )
            
            response = self.gemini_model.generate_content(prompt)
            self.api_calls["gemini_api"] += 1
            
            analysis_text = response.text
            analysis_length = len(analysis_text)
            
            processing_time = time.time() - start_time
            
            print(f"✅ Chapter analysis completed ({analysis_length} characters)")
            print(f"⏱️ Processing time: {processing_time:.2f} seconds")
            print(f"📊 API call #{self.api_calls['gemini_api']} - Chapter Analysis")
            print(f"📊 Total Gemini API calls so far: {self.api_calls['gemini_api']}")
            
            return {
                "chapter_summary": analysis_text,
                "analyzed_at": datetime.now().isoformat()
            }
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Error in chapter analysis: {str(e)[:100]}...")
            print(f"⏱️ Failed after: {processing_time:.2f} seconds")
            
            return {
                "chapter_summary": "અધ્યાય વિશ્લેષણમાં ભૂલ",
                "analyzed_at": datetime.now().isoformat()
            }
    
    def extract_topics_from_analysis(self, chapter_info):
        """Extract topics list from chapter analysis"""
        print("\n" + "="*50)
        print("📋 STEP 6: TOPIC EXTRACTION")
        print("="*50)
        
        analysis = chapter_info["chapter_summary"]
        print(f"📊 Analyzing {len(analysis)} characters of chapter summary")
        
        topics = []
        lines = analysis.split('\n')
        
        print("🔍 Searching for topics in chapter analysis...")
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('- ')):
                topic = line.lstrip('0123456789.- ').strip()
                if topic and len(topic) > 5:
                    topics.append(topic)
                    print(f"  ✅ Topic {len(topics)}: {topic[:50]}...")
        
        print(f"\n📋 TOPIC EXTRACTION COMPLETED")
        print(f"🎯 Topics extracted: {len(topics)}")
        
        if len(topics) == 0:
            print("⚠️ No topics found in chapter analysis. Creating default topics...")
            topics = ["સામાન્ય ગણિતીય સંકલ્પનાઓ", "સૂત્રો અને ગણતરી", "ઉદાહરણો અને કસોટીઓ"]
        
        print("📊 No API calls required for topic extraction")
        
        print("\n📋 EXTRACTED TOPICS:")
        for i, topic in enumerate(topics, 1):
            print(f"  {i}. {topic}")
        
        return topics
    
    def assign_topics_to_pages(self, pages_data, topics_list):
        """Assign non-overlapping topics to each page"""
        print("\n" + "="*50)
        print("🏷️ STEP 7: TOPIC ASSIGNMENT")
        print("="*50)
        
        print(f"📄 Pages to process: {len(pages_data)}")
        print(f"🎯 Available topics: {len(topics_list)}")
        print(f"📊 Current API calls - Vision: {self.api_calls['vision_api']}, Gemini: {self.api_calls['gemini_api']}")
        
        topics_with_numbers = []
        for i, topic in enumerate(topics_list, 1):
            topics_with_numbers.append(f"{i}. {topic}")
        
        topics_string = "\n".join(topics_with_numbers)
        print(f"📋 Topics formatted for AI assignment")
        
        start_time = time.time()
        successful_assignments = 0
        failed_assignments = 0
        total_topic_assignments = 0
        
        for page in tqdm(pages_data, desc="🏷️ Assigning topics"):
            try:
                print(f"\n📄 Processing Page {page['page_number']}")
                
                text_length = len(page["text"])
                text_sample = page["text"][:2000]
                
                print(f"  📊 Text length: {text_length} characters (using first 2000)")
                
                prompt = self.prompts["topic_assignment"].format(
                    page_text=text_sample,
                    topics_list=topics_string
                )
                
                print(f"  🔄 Sending to Gemini API...")
                response = self.gemini_model.generate_content(prompt)
                
                self.api_calls["gemini_api"] += 1
                successful_assignments += 1
                
                topic_numbers = []
                response_text = response.text.strip()
                
                print(f"  📝 AI Response: {response_text[:50]}...")
                
                for part in response_text.split(','):
                    try:
                        num = int(part.strip())
                        if 1 <= num <= len(topics_list) and num not in topic_numbers:
                            topic_numbers.append(num)
                    except ValueError:
                        continue
                
                page["assigned_topics"] = topic_numbers
                page["topics_assigned_at"] = datetime.now().isoformat()
                
                total_topic_assignments += len(topic_numbers)
                print(f"  ✅ Assigned {len(topic_numbers)} topics: {topic_numbers} - API call #{self.api_calls['gemini_api']}")
                
                time.sleep(1)
            except Exception as e:
                failed_assignments += 1
                print(f"  ❌ Error: {str(e)[:100]}...")
                page["assigned_topics"] = []
        
        processing_time = time.time() - start_time
        avg_topics_per_page = total_topic_assignments / len(pages_data) if pages_data else 0
        
        print("\n✅ TOPIC ASSIGNMENT COMPLETED")
        print(f"📄 Successful assignments: {successful_assignments}")
        print(f"❌ Failed assignments: {failed_assignments}")
        print(f"🎯 Total topic assignments: {total_topic_assignments}")
        print(f"📊 Average topics per page: {avg_topics_per_page:.1f}")
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"📊 Gemini API calls for topic assignment: {successful_assignments}")
        print(f"📊 Total Gemini API calls so far: {self.api_calls['gemini_api']}")
        
        return pages_data
    
    def save_results(self, pdf_path, pages_data, chapter_info, topics_list):
        """Save all results to single JSON file"""
        print("\n" + "="*50)
        print("💾 STEP 8: SAVE RESULTS")
        print("="*50)
        
        processing_end_time = datetime.now()
        total_processing_time = (processing_end_time - self.api_calls["start_time"]).total_seconds()
        
        total_images = sum(len(page.get('images', [])) for page in pages_data)
        total_characters = sum(len(page.get('text', '')) for page in pages_data)
        total_topic_assignments = sum(len(page.get('assigned_topics', [])) for page in pages_data)
        
        print(f"📊 FINAL PROCESSING STATISTICS:")
        print(f"  📄 Total pages: {len(pages_data)}")
        print(f"  🖼️ Total images: {total_images}")
        print(f"  📝 Total characters extracted: {total_characters:,}")
        print(f"  🎯 Total topics extracted: {len(topics_list)}")
        print(f"  🏷️ Total topic assignments: {total_topic_assignments}")
        print(f"  ⏱️ Total processing time: {total_processing_time:.2f} seconds")
        
        output_data = {
            "metadata": {
                "source_pdf": os.path.basename(pdf_path),
                "board": "GSEB",
                "class": 10,
                "subject": "Mathematics",
                "medium": "Gujarati",
                "processed_at": processing_end_time.isoformat(),
                "total_pages": len(pages_data),
                "total_topics": len(topics_list),
                "total_images": total_images,
                "total_characters": total_characters,
                "processing_time_seconds": round(total_processing_time, 2),
                "api_usage": {
                    "vision_api_calls": self.api_calls["vision_api"],
                    "gemini_api_calls": self.api_calls["gemini_api"],
                    "total_api_calls": self.api_calls["vision_api"] + self.api_calls["gemini_api"]
                }
            },
            "chapter_info": {
                **chapter_info,
                "extracted_topics": topics_list
            },
            "pages": pages_data
        }
        
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        timestamp = processing_end_time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"gseb_class10_maths_{base_name}_{timestamp}.json"
        
        print(f"📁 Saving to file: {output_filename}")
        
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            file_size = os.path.getsize(output_filename) / (1024 * 1024)
            
            print(f"✅ SAVE COMPLETED")
            print(f"💾 File size: {file_size:.2f} MB")
            print(f"📊 API USAGE SUMMARY:")
            print(f"  🔍 Google Vision API calls: {self.api_calls['vision_api']}")
            print(f"  🧠 Google Gemini API calls: {self.api_calls['gemini_api']}")
            print(f"  📊 Total API calls: {self.api_calls['vision_api'] + self.api_calls['gemini_api']}")
            
            return output_filename
        except Exception as e:
            print(f"❌ Error saving file: {str(e)}")
            return None
    
    def process_pdf(self, pdf_path):
        """Main processing pipeline with comprehensive logging and API counting"""
        print("\n" + "="*70)
        print("🚀 GSEB PDF PROCESSING PIPELINE STARTED")
        print("="*70)
        print(f"📄 File: {os.path.basename(pdf_path)}")
        print(f"📚 Target: Class 10 Mathematics (Gujarati Medium)")
        print(f"🕐 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Initial API counters - Vision: 0, Gemini: 0")
        
        try:
            # Step 1: Extract PDF pages with images
            pages_data = self.extract_pdf_with_images(pdf_path)
            
            if not pages_data:
                print("\n❌ PIPELINE FAILED: No pages extracted")
                return None
            
            # Step 2: Generate AI descriptions for images/diagrams
            pages_data = self.describe_images_with_ai(pages_data)
            
            # Step 3: Integrate image references into text
            pages_data = self.integrate_images_in_text(pages_data)
            
            # Step 4: Summarize each page (with image context)
            pages_data = self.summarize_pages(pages_data)
            
            # Step 5: Analyze complete chapter
            chapter_info = self.analyze_chapter(pages_data)
            
            # Step 6: Extract topics from analysis
            topics_list = self.extract_topics_from_analysis(chapter_info)
            
            # Step 7: Assign topics to pages
            pages_data = self.assign_topics_to_pages(pages_data, topics_list)
            
            # Step 8: Save complete results
            output_file = self.save_results(pdf_path, pages_data, chapter_info, topics_list)
            
            if not output_file:
                print("\n❌ PIPELINE FAILED: Could not save results")
                return None
            
            total_images = sum(len(page.get('images', [])) for page in pages_data)
            total_characters = sum(len(page.get('text', '')) for page in pages_data)
            
            try:
                total_processing_time = (datetime.now() - self.api_calls["start_time"]).total_seconds()
                vision_calls = self.api_calls['vision_api']
                gemini_calls = self.api_calls['gemini_api']
            except (AttributeError, KeyError):
                total_processing_time = 0
                vision_calls = 0
                gemini_calls = 0
            
            print("\n" + "="*70)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"📊 FINAL STATISTICS:")
            print(f"  📄 Pages processed: {len(pages_data)}")
            print(f"  🖼️ Images processed: {total_images}")
            print(f"  📝 Characters extracted: {total_characters:,}")
            print(f"  🎯 Topics extracted: {len(topics_list)}")
            if total_processing_time > 0:
                print(f"  ⏱️ Total processing time: {total_processing_time:.2f} seconds")
            print(f"📊 API USAGE:")
            print(f"  🔍 Google Vision API calls: {vision_calls}")
            print(f"  🧠 Google Gemini API calls: {gemini_calls}")
            print(f"  📊 Total API calls: {vision_calls + gemini_calls}")
            print(f"💾 Output file: {output_file}")
            print("="*70)
            
            return output_file
        except Exception as e:
            try:
                total_processing_time = (datetime.now() - self.api_calls["start_time"]).total_seconds()
                vision_calls = self.api_calls['vision_api']
                gemini_calls = self.api_calls['gemini_api']
            except (AttributeError, KeyError):
                total_processing_time = 0
                vision_calls = 0
                gemini_calls = 0
            
            print("\n" + "="*70)
            print("❌ PIPELINE FAILED!")
            print("="*70)
            print(f"🚫 Error: {str(e)}")
            if total_processing_time > 0:
                print(f"⏱️ Failed after: {total_processing_time:.2f} seconds")
            print(f"📊 API calls made before failure:")
            print(f"  🔍 Google Vision API: {vision_calls}")
            print(f"  🧠 Google Gemini API: {gemini_calls}")
            print("="*70)
            
            return None

def main():
    """Main execution function"""
    print("📚 GSEB Question Paper Generation System")
    print("🔧 PDF Processing Module - Class 10 Maths (Gujarati)")
    print("✨ With Image Recognition & Educational Descriptions")
    print("=" * 70)
    
    # Initialize processor
    try:
        processor = GSEBPDFProcessor()
        
    except ValueError as e:
        print(f"❌ Initialization Error: {str(e)}")
        print("Please check your .env file and API keys.")
        return
    
    # Get PDF file path from user
    pdf_path = input("\n📁 Enter the path to your Class 10 Maths Gujarati PDF: ").strip()
    
    if not os.path.exists(pdf_path):
        print("❌ File not found! Please check the path.")
        return
    
    processor.current_chapter = "દ્વિચલ સુરેખ સમીકરણયુગ્મ"
    # Process the PDF
    result = processor.process_pdf(pdf_path)
    
    if result:
        print(f"\n✅ SUCCESS! Check the output file: {result}")
        print("\n📖 The file contains:")
        print("   • Page-by-page text extraction")  
        print("   • Educational image descriptions")
        print("   • Comprehensive page summaries")
        print("   • Complete chapter analysis")
        print("   • Topic extraction and assignment")
    else:
        print("\n❌ Processing failed! Check the error messages above.")

if __name__ == "__main__":
    main()