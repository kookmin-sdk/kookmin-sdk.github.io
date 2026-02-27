/**
 * Auto Translation Feature
 * Enables automatic translation of Korean content to English and vice versa
 * Currently uses MyMemory API (free, no key required)
 * 
 * For better translation quality, you can upgrade to Google Translate API:
 * - Get API key from: https://cloud.google.com/docs/authentication/getting-started
 * - Update the translateWithGoogle function below
 * - Set USE_GOOGLE_TRANSLATE = true
 */

const USE_GOOGLE_TRANSLATE = false;  // Change to true if using Google Translate API
const GOOGLE_API_KEY = 'YOUR_GOOGLE_API_KEY';  // Replace with your actual API key

class PageTranslator {
    constructor() {
        this.currentLanguage = 'ko';
        this.translations = new Map();
        this.mymemoryApiUrl = 'https://api.mymemory.translated.net/get';
        this.googleApiUrl = 'https://translation.googleapis.com/language/translate/v2';
        this.setupTranslationButton();
        this.observePageChanges();
    }

    setupTranslationButton() {
        // Create translation toggle button if it doesn't exist
        const existingBtn = document.getElementById('toggle-translation');
        if (!existingBtn) {
            const btn = document.createElement('button');
            btn.id = 'toggle-translation';
            btn.className = 'translation-toggle';
            btn.innerHTML = '🌐 English';
            btn.title = 'Click to toggle between Korean and English';
            btn.style.cssText = `
                position: fixed;
                top: 80px;
                right: 20px;
                padding: 10px 18px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                z-index: 1000;
                font-size: 14px;
                font-weight: 500;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            `;
            
            btn.addEventListener('mouseover', () => {
                btn.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)';
                btn.style.transform = 'translateY(-2px)';
            });
            
            btn.addEventListener('mouseout', () => {
                btn.style.boxShadow = '0 2px 10px rgba(0,0,0,0.2)';
                btn.style.transform = 'translateY(0)';
            });
            
            btn.addEventListener('click', () => this.toggleLanguage());
            
            document.body.appendChild(btn);
        }
    }

    toggleLanguage() {
        if (this.currentLanguage === 'ko') {
            this.currentLanguage = 'en';
            this.translatePage('ko', 'en');
            document.getElementById('toggle-translation').innerHTML = '🇰🇷 한국어';
        } else {
            this.currentLanguage = 'ko';
            this.restoreOriginal();
            document.getElementById('toggle-translation').innerHTML = '🌐 English';
        }
    }

    async translatePage(sourceLang, targetLang) {
        const btn = document.getElementById('toggle-translation');
        btn.disabled = true;
        btn.innerHTML = '⏳ 번역 중...';

        // Target elements for translation
        const selectors = [
            '#libdoc-page-title',
            '#libdoc-content h1',
            '#libdoc-content h2',
            '#libdoc-content h3',
            '#libdoc-content h4',
            '#libdoc-content p',
            '#libdoc-content li',
            '#libdoc-content td',
            '#libdoc-content th',
            '.post-content'
        ];

        const elements = document.querySelectorAll(selectors.join(', '));
        let translatedCount = 0;

        for (const element of elements) {
            // Skip code blocks and script tags
            if (element.tagName === 'CODE' || element.tagName === 'SCRIPT' || element.tagName === 'STYLE') {
                continue;
            }

            // Get text content, excluding child elements
            let text = element.textContent?.trim();
            
            if (text && text.length > 0 && !this.isEnglish(text) && !element.dataset.originalText) {
                try {
                    element.dataset.originalText = text;
                    
                    if (USE_GOOGLE_TRANSLATE) {
                        var translatedText = await this.translateWithGoogle(text, targetLang);
                    } else {
                        var translatedText = await this.translateText(text, sourceLang, targetLang);
                    }
                    
                    // Only update if translation was successful and different
                    if (translatedText && translatedText !== text) {
                        element.textContent = translatedText;
                        translatedCount++;
                    }
                } catch (error) {
                    console.error('Translation error for:', text, error);
                }
            }

            // Add small delay to avoid rate limiting
            if (translatedCount % 5 === 0) {
                await new Promise(resolve => setTimeout(resolve, 100));
            }
        }

        btn.disabled = false;
        btn.innerHTML = '🇰🇷 한국어';
    }

    restoreOriginal() {
        const elements = document.querySelectorAll('[data-original-text]');
        elements.forEach(element => {
            if (element.dataset.originalText) {
                element.textContent = element.dataset.originalText;
            }
        });
    }

    async translateText(text, sourceLang, targetLang) {
        // Use caching to avoid hitting API limits
        const cacheKey = `${sourceLang}-${targetLang}-${text.substring(0, 100)}`;
        
        if (this.translations.has(cacheKey)) {
            return this.translations.get(cacheKey);
        }

        try {
            const response = await fetch(
                `${this.mymemoryApiUrl}?q=${encodeURIComponent(text)}&langpair=${sourceLang}|${targetLang}`,
                { method: 'GET' }
            );

            const data = await response.json();

            if (data.responseStatus === 200 && data.responseData.translatedText) {
                const translatedText = data.responseData.translatedText;
                this.translations.set(cacheKey, translatedText);
                return translatedText;
            } else {
                console.warn('Translation API response:', data);
                return text;
            }
        } catch (error) {
            console.error('MyMemory API error:', error);
            return text;
        }
    }

    async translateWithGoogle(text, targetLanguage) {
        // For production, use official Google Cloud Translation API
        // This requires authentication and API key setup
        
        // Fallback to MyMemory if API key not set
        if (!GOOGLE_API_KEY || GOOGLE_API_KEY === 'YOUR_GOOGLE_API_KEY') {
            return await this.translateText(text, 'ko', targetLanguage);
        }

        try {
            const response = await fetch(this.googleApiUrl, {
                method: 'POST',
                body: JSON.stringify({
                    q: text,
                    target: targetLanguage,
                    source: 'ko',
                    key: GOOGLE_API_KEY
                }),
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();
            
            if (data.data && data.data.translations && data.data.translations[0]) {
                return data.data.translations[0].translatedText;
            } else {
                console.warn('Google Translate error:', data);
                return text;
            }
        } catch (error) {
            console.error('Google Translate API error:', error);
            // Fallback to MyMemory
            return await this.translateText(text, 'ko', targetLanguage);
        }
    }

    isEnglish(text) {
        // Check if text contains mostly English characters
        // Korean Hangul characters: \uac00-\ud7a3
        const koreanRegex = /[\uac00-\ud7a3]/g;
        const koreanMatches = text.match(koreanRegex) || [];
        
        // If more than 30% Korean characters, it's Korean
        return koreanMatches.length < text.length * 0.3;
    }

    observePageChanges() {
        // If page content changes dynamically, update translator
        const observer = new MutationObserver(() => {
            // Could refresh translator state here if needed
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
}

// Initialize translator when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.pageTranslator = new PageTranslator();
    });
} else {
    window.pageTranslator = new PageTranslator();
}
