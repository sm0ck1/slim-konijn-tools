import stanza
import nltk
import os
import gzip
import shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from nltk.corpus import wordnet
import fasttext
import requests
from pathlib import Path

app = FastAPI(title="Slim Konijn Tools")

# Download NLTK data if not present
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')


# --- FastText Model Download Helper ---
def download_fasttext_model(lang: str) -> str:
    """Download FastText model if not exists"""
    model_path = f"cc.{lang}.300.bin"
    
    if os.path.exists(model_path):
        print(f"FastText {lang} model already exists at {model_path}")
        return model_path
    
    gz_path = f"{model_path}.gz"
    url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{gz_path.split('/')[-1]}"
    
    print(f"Downloading FastText {lang} model from {url}...")
    print("This may take a while (model is ~6.8GB compressed)...")
    
    try:
        # Download with progress
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(gz_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = (downloaded / total_size) * 100
                    print(f"Download progress: {percent:.1f}%", end='\r')
        
        print(f"\nDownload complete. Extracting {gz_path}...")
        
        # Extract with better error handling
        try:
            with gzip.open(gz_path, 'rb') as f_in:
                with open(model_path, 'wb') as f_out:
                    # Copy in chunks and show progress
                    chunk_size = 8192
                    while True:
                        chunk = f_in.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
            
            print(f"Extraction complete. Verifying...")
            
            # Verify extracted file size
            extracted_size = os.path.getsize(model_path)
            print(f"Extracted file size: {extracted_size / (1024**3):.2f} GB")
            
            if extracted_size < 1_000_000_000:  # Less than 1GB
                raise Exception(f"Extracted file too small: {extracted_size / (1024**2):.2f} MB")
            
            # Remove .gz file only after successful extraction
            os.remove(gz_path)
            print(f"FastText {lang} model ready at {model_path}")
            return model_path
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(model_path):
                os.remove(model_path)
            raise Exception(f"Extraction failed: {e}")
        
    except Exception as e:
        print(f"Error downloading FastText model: {e}")
        if os.path.exists(gz_path):
            os.remove(gz_path)
        raise


# --- Load FastText Models ---
def load_fasttext_model(lang: str):
    """Load or download FastText model"""
    try:
        model_path = download_fasttext_model(lang)
        print(f"Loading FastText {lang} model...")
        
        # Check file size to ensure it was downloaded correctly
        file_size = os.path.getsize(model_path)
        print(f"Model file size: {file_size / (1024**3):.2f} GB")
        
        if file_size < 1_000_000_000:  # Less than 1GB, probably corrupted
            print(f"Warning: Model file seems too small ({file_size / (1024**2):.2f} MB), re-downloading...")
            os.remove(model_path)
            model_path = download_fasttext_model(lang)
        
        # Load using official fasttext library
        model = fasttext.load_model(model_path)
        print(f"FastText {lang} model loaded successfully")
        return model
    except Exception as e:
        print(f"Could not load FastText {lang} model: {e}")
        import traceback
        traceback.print_exc()
        return None


# Initialize Stanza models for English and Dutch
nlp_en = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', use_gpu=False)
nlp_nl = stanza.Pipeline(lang='nl', processors='tokenize,pos,lemma,depparse', use_gpu=False)

# Initialize FastText models
print("Initializing FastText models...")
model_en = load_fasttext_model('en')
model_nl = load_fasttext_model('nl')


class WordAnalysis(BaseModel):
    word: str
    pos_tag: str
    lemma: str
    synonyms: List[str] = []
    is_valid: bool = True
    suggestions: List[str] = []


class SentenceAnalysis(BaseModel):
    sentence: str
    is_valid: bool
    words: List[WordAnalysis]
    tokens: List[str]


class AnalyzeWordRequest(BaseModel):
    word: str
    language: str = "en"


class AnalyzeSentenceRequest(BaseModel):
    sentence: str
    language: str = "en"
    include_synonyms: bool = True
    synonyms_limit: int = 5


class AnalyzeBatchRequest(BaseModel):
    sentences: List[str]
    language: str = "en"
    include_synonyms: bool = True
    synonyms_limit: int = 5


class SynonymsRequest(BaseModel):
    word: str
    language: str = "en"
    topn: int = 10


class SynonymsResponse(BaseModel):
    word: str
    lemma: str
    synonyms: List[str]
    source: str


# --- Helper functions for synonyms ---

def lemmatize(word: str, lang: str = 'en') -> str:
    """Lemmatize a word using Stanza"""
    nlp = nlp_en if lang == 'en' else nlp_nl
    doc = nlp(word)
    if doc.sentences and doc.sentences[0].words:
        return doc.sentences[0].words[0].lemma
    return word


def check_word_validity(word: str, lang: str = 'en') -> bool:
    """Check if word exists in FastText vocabulary"""
    model = model_en if lang == 'en' else model_nl
    if model is None:
        return True  # If model not loaded, assume valid
    
    # Try word as-is and lowercase
    word_lower = word.lower()
    lemma = lemmatize(word, lang).lower()
    
    # FastText can generate vectors for any word, so we check by getting neighbors
    # If it finds good neighbors, the word is likely valid
    try:
        # Try to get neighbors - if it works, word is valid enough
        neighbors = model.get_nearest_neighbors(word_lower, k=1)
        # If closest neighbor is very close (>0.8), word is likely valid
        if neighbors and neighbors[0][0] > 0.8:
            return True
        # Also try lemma
        neighbors_lemma = model.get_nearest_neighbors(lemma, k=1)
        return neighbors_lemma and neighbors_lemma[0][0] > 0.8
    except:
        return False


def find_similar_words(word: str, lang: str = 'en', topn: int = 5) -> List[str]:
    """Find similar words for spell correction using FastText"""
    model = model_en if lang == 'en' else model_nl
    if model is None:
        return []
    
    word_lower = word.lower()
    
    try:
        # FastText get_nearest_neighbors returns list of (similarity, word) tuples
        neighbors = model.get_nearest_neighbors(word_lower, k=topn)
        # Filter by similarity threshold and return only words
        return [w for sim, w in neighbors if sim > 0.6]
    except Exception as e:
        print(f"Error finding similar words for '{word}': {e}")
        return []


def get_synonyms_en(word: str) -> List[str]:
    """Get synonyms for English word using WordNet"""
    lemma = lemmatize(word, 'en')
    synonyms = set()
    
    for syn in wordnet.synsets(lemma):
        for lemma_obj in syn.lemmas():
            # Replace underscores with spaces
            syn_word = lemma_obj.name().replace('_', ' ')
            synonyms.add(syn_word)
    
    # Remove the original word from synonyms
    synonyms.discard(word.lower())
    synonyms.discard(lemma.lower())
    
    return sorted(list(synonyms))


def get_synonyms_nl(word: str, topn: int = 10) -> List[str]:
    """Get synonyms for Dutch word using FastText"""
    if model_nl is None:
        return []
    
    lemma = lemmatize(word, 'nl')
    try:
        # FastText get_nearest_neighbors returns list of (similarity, word) tuples
        neighbors = model_nl.get_nearest_neighbors(lemma.lower(), k=topn)
        return [w for sim, w in neighbors if w.lower() != lemma.lower()]
    except Exception:
        # If error with lemma, try original word
        try:
            neighbors = model_nl.get_nearest_neighbors(word.lower(), k=topn)
            return [w for sim, w in neighbors if w.lower() != word.lower()]
        except Exception:
            return []


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze/word")
async def analyze_word(request: AnalyzeWordRequest) -> Dict[str, Any]:
    """Analyze a single word for POS tag and lemma"""
    nlp = nlp_en if request.language == "en" else nlp_nl
    doc = nlp(request.word)
    
    if doc.sentences and doc.sentences[0].words:
        word_obj = doc.sentences[0].words[0]
        return {
            "word": request.word,
            "pos_tag": word_obj.pos,
            "lemma": word_obj.lemma
        }
    
    return {"word": request.word, "pos_tag": None, "lemma": None}


@app.post("/analyze/sentence")
async def analyze_sentence(request: AnalyzeSentenceRequest) -> SentenceAnalysis:
    """Analyze a sentence for structure, words, POS tags, lemmas and synonyms"""
    try:
        nlp = nlp_en if request.language == "en" else nlp_nl
        doc = nlp(request.sentence)
        
        words_analysis = []
        tokens = []
        
        for sent in doc.sentences:
            for word in sent.words:
                tokens.append(word.text)
                
                # Check word validity using FastText
                is_valid = check_word_validity(word.text, request.language)
                
                # Find suggestions if word is invalid
                suggestions = []
                if not is_valid:
                    suggestions = find_similar_words(word.text, request.language, topn=5)
                
                # Get synonyms for each word if requested
                synonyms = []
                if request.include_synonyms:
                    if request.language == "en":
                        synonyms = get_synonyms_en(word.text)[:request.synonyms_limit]
                    elif request.language == "nl":
                        synonyms = get_synonyms_nl(word.text, topn=request.synonyms_limit)
                
                words_analysis.append(
                    WordAnalysis(
                        word=word.text,
                        pos_tag=word.pos,
                        lemma=word.lemma,
                        synonyms=synonyms,
                        is_valid=is_valid,
                        suggestions=suggestions
                    )
                )
        
        # Grammar validation using dependency parsing
        # Check if sentence has a ROOT (main verb) and proper structure
        has_root = False
        has_subject = False
        
        for sent in doc.sentences:
            for word in sent.words:
                # ROOT is the main verb/predicate
                if word.deprel == "root":
                    has_root = True
                # nsubj is the subject (nominal subject)
                if word.deprel in ["nsubj", "nsubj:pass"]:
                    has_subject = True
            
            print(f"DEBUG - Dependencies: {[(w.text, w.deprel, w.head) for w in sent.words]}")
        
        # Sentence is valid if it has ROOT (main predicate) 
        # and ideally should have subject, but some imperative sentences don't have explicit subject
        is_valid = has_root
        
        return SentenceAnalysis(
            sentence=request.sentence,
            is_valid=is_valid,
            words=words_analysis,
            tokens=tokens
        )
    except Exception as e:
        return SentenceAnalysis(
            sentence=request.sentence,
            is_valid=False,
            words=[],
            tokens=[]
        )


@app.post("/analyze/batch")
async def analyze_batch(request: AnalyzeBatchRequest) -> List[SentenceAnalysis]:
    """Analyze multiple sentences"""
    results = []
    for sentence in request.sentences:
        result = await analyze_sentence(
            AnalyzeSentenceRequest(
                sentence=sentence,
                language=request.language,
                include_synonyms=request.include_synonyms,
                synonyms_limit=request.synonyms_limit
            )
        )
        results.append(result)
    return results


@app.post("/synonyms")
async def get_synonyms(request: SynonymsRequest) -> SynonymsResponse:
    """Get synonyms for a word in English or Dutch"""
    try:
        if request.language == "en":
            lemma = lemmatize(request.word, 'en')
            synonyms = get_synonyms_en(request.word)
            source = "WordNet"
        elif request.language == "nl":
            lemma = lemmatize(request.word, 'nl')
            synonyms = get_synonyms_nl(request.word, topn=request.topn)
            source = "FastText" if model_nl else "None"
            
            if not synonyms and model_nl is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"FastText model not available. Please download cc.nl.300.bin to get Dutch synonyms."
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {request.language}. Use 'en' or 'nl'."
            )
        
        return SynonymsResponse(
            word=request.word,
            lemma=lemma,
            synonyms=synonyms,
            source=source
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
