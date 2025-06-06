import requests
from bs4 import BeautifulSoup
from googlesearch import search
import re
import json
import os
import wikipedia

# 1. Trusted websites (Türkçe siteler eklendi)
TRUSTED_SITES = [
    "bbc.com", "reuters.com", "nytimes.com", "anadoluajansi.com.tr",
    "dw.com", "cnn.com", "forbes.com", "who.int", "healthline.com",
    "snopes.com", "politifact.com", "hurriyet.com.tr", "sozcu.com.tr",
    "ntv.com.tr", "cnnturk.com", "sabah.com.tr", "milliyet.com.tr",
    "haberturk.com", "ensonhaber.com", "teyit.org", "dogrulukpayi.com"
]

# 2. Cache for faster re-checks
EXPLANATION_CACHE_FILE = "explanations.json"
if os.path.exists(EXPLANATION_CACHE_FILE):
    with open(EXPLANATION_CACHE_FILE, "r", encoding="utf-8") as f:
        EXPLANATION_CACHE = json.load(f)
else:
    EXPLANATION_CACHE = {}

# 3. Türkçe açıklamalar
TURKISH_EXPLANATIONS = {
    "no_source": "Bu iddia güvenilir kaynaklarda desteklenmemektedir. Bilimsel konsensüs veya güvenilir bilgi eksikliği nedeniyle bu açıklama muhtemelen yanlış veya yanıltıcıdır.",
    "analysis_complete": "Yapay zeka analizi tamamlandı. Metnin güvenilirlik oranı değerlendirildi.",
    "no_reliable_info": "Bu konu hakkında güvenilir kaynaklarda yeterli bilgi bulunamadı.",
    "contradiction_found": "Bu iddia güvenilir kaynaklarda çürütülmüştür.",
    "supported_claim": "Bu iddia güvenilir kaynaklarda desteklenmektedir."
}

# 4. Helper functions
def is_trusted_site(url):
    return any(site in url for site in TRUSTED_SITES)

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def search_google_turkish(query, max_results=5):
    """Türkçe öncelikli Google araması"""
    try:
        # Önce Türkçe sitelerden ara
        turkish_query = f"{query} site:tr OR site:com.tr"
        results = list(search(turkish_query, num_results=max_results, lang="tr"))
        
        # Eğer Türkçe sonuç yoksa genel arama yap
        if not results:
            results = list(search(query, num_results=max_results))
        
        return results
    except Exception as e:
        print(f"[!] Google search error: {e}")
        return []

def translate_to_turkish(text):
    """Basit İngilizce-Türkçe çeviri (anahtar kelimeler için)"""
    translations = {
        "false": "yanlış", "not true": "doğru değil", "myth": "efsane", 
        "debunked": "çürütülmüş", "fake": "sahte", "true": "doğru",
        "verified": "doğrulanmış", "confirmed": "onaylanmış",
        "studies show": "araştırmalar gösteriyor", "research indicates": "araştırmalar gösteriyor",
        "according to": "göre", "experts say": "uzmanlar diyor",
        "scientists": "bilim insanları", "study": "çalışma", "research": "araştırma"
    }
    
    turkish_text = text
    for eng, tr in translations.items():
        turkish_text = re.sub(rf'\b{eng}\b', tr, turkish_text, flags=re.IGNORECASE)
    
    return turkish_text

def extract_best_paragraph(content, keywords):
    paragraphs = content.split("\n")
    scored = []
    for para in paragraphs:
        if len(para.strip()) < 50:  # Çok kısa paragrafları atla
            continue
        score = sum(1 for k in keywords if k in para.lower())
        if score:
            scored.append((score, para))
    if scored:
        best_para = sorted(scored, reverse=True)[0][1]
        # Türkçe çeviri uygula
        return translate_to_turkish(clean_text(best_para))
    return ""

def extract_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'tr-TR,tr;q=0.9,en;q=0.8'
        }
        response = requests.get(url, timeout=10, headers=headers)
        response.encoding = 'utf-8'
        
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.get_text().strip() if soup.title else ""
        
        # Script ve style etiketlerini temizle
        for script in soup(["script", "style", "nav", "footer", "aside"]):
            script.decompose()
            
        paragraphs = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 30]
        content = "\n".join(paragraphs[:15])  # ilk 15 paragraf
        
        return title.lower(), content.lower(), content
    except Exception as e:
        print(f"[!] Error reading {url}: {e}")
        return "", "", ""

# 5. Wikipedia Türkçe desteği
def search_wikipedia_turkish(news_text):
    try:
        # Önce Türkçe Wikipedia'yı dene
        wikipedia.set_lang("tr")
        try:
            summary = wikipedia.summary(news_text, sentences=3)
            if summary:
                return clean_text(summary)
        except:
            pass
        
        # Türkçe bulamazsa İngilizce dene ve çevir
        wikipedia.set_lang("en")
        summary = wikipedia.summary(news_text, sentences=2)
        return translate_to_turkish(clean_text(summary))
        
    except Exception as e:
        print(f"[!] Wikipedia error: {e}")
        return None

# 6. Main explanation logic - Türkçe
def get_explanation(news_text):
    if news_text in EXPLANATION_CACHE:
        return EXPLANATION_CACHE[news_text]

    keywords = set(re.findall(r'\b\w{4,}\b', news_text.lower()))
    final = None

    # Phase 1: Türkçe kaynaklarda destek ara
    print("🔍 Türkçe kaynaklarda aranıyor...")
    search_results = search_google_turkish(news_text)
    
    for url in search_results:
        if not is_trusted_site(url):
            continue
        print(f"✓ Güvenilir kaynak bulundu: {url}")
        title, content_lc, original_content = extract_content(url)
        best_para = extract_best_paragraph(original_content, keywords)
        if best_para and len(best_para) > 50:
            final = {
                "explanation": f"{TURKISH_EXPLANATIONS['supported_claim']} {best_para}",
                "source": url
            }
            break

    # Phase 2: Çelişki kontrolü
    if not final:
        print("🔍 Çelişki kontrolü yapılıyor...")
        contradiction_queries = [
            f"{news_text} yanlış",
            f"{news_text} doğru mu",
            f"{news_text} teyit",
            f"false {news_text}"
        ]
        
        for query in contradiction_queries:
            for url in search_google_turkish(query, 3):
                if not is_trusted_site(url):
                    continue
                title, content_lc, original_content = extract_content(url)
                contradiction_keywords = ["yanlış", "doğru değil", "efsane", "çürütülmüş", "false", "debunked"]
                if any(w in title for w in contradiction_keywords) or any(w in content_lc for w in contradiction_keywords):
                    best_para = extract_best_paragraph(original_content, contradiction_keywords)
                    if best_para:
                        final = {
                            "explanation": f"{TURKISH_EXPLANATIONS['contradiction_found']} {best_para}",
                            "source": url
                        }
                        break
            if final:
                break

    # Phase 3: Wikipedia Türkçe fallback
    if not final:
        print("🔍 Wikipedia'da aranıyor...")
        wiki_summary = search_wikipedia_turkish(news_text)
        if wiki_summary and len(wiki_summary) > 30:
            final = {
                "explanation": f"{TURKISH_EXPLANATIONS['analysis_complete']} {wiki_summary}",
                "source": "Wikipedia"
            }

    # Phase 4: Son çare - Türkçe varsayılan mesaj
    if not final:
        final = {
            "explanation": TURKISH_EXPLANATIONS["no_source"],
            "source": "Yapay Zeka Analizi"
        }

    # Cache'e kaydet
    EXPLANATION_CACHE[news_text] = final
    with open(EXPLANATION_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(EXPLANATION_CACHE, f, indent=2, ensure_ascii=False)

    return final

# 7. Test fonksiyonu
if __name__ == "__main__":
    test_claim = "Koronavirüs aşısı zararlıdır"
    result = get_explanation(test_claim)
    print("\n[Türkçe Açıklama]")
    print(f"Kaynak: {result['source']}")
    print(f"Açıklama: {result['explanation']}")