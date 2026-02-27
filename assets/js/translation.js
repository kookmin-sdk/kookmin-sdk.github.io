/**
 * Auto Translation Feature
 * Enables automatic translation of Korean content to English using MyMemory Translation API
 * 
 * Usage:
 * - Add data-translatable="true" to elements you want to translate
 * - Add a button with id="toggle-translation" to trigger translation
 */

class PageTranslator {
    constructor() {
        this.currentLanguage = 'ko';
        this.translations = new Map();
        this.translationApiUrl = 'https://api.mymemory.translated.net/get';
        this.setupTranslationButton();
    }

    setupTranslationButton() {
        // Create translation toggle button if it doesn't exist
        const existingBtn = document.getElementById('toggle-translation');
        if (!existingBtn) {
            const btn = document.createElement('button');
            btn.id = 'toggle-translation';
            btn.className = 'translation-toggle';
            btn.innerHTML = '🌐 English';
            btn.style.cssText = `
                position: fixed;
                top: 80px;
                right: 20px;
                padding: 8px 16px;
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
            this.translatePage();
            document.getElementById('toggle-translation').innerHTML = '🇰🇷 한국어';
        } else {
            this.currentLanguage = 'ko';
            this.restoreOriginal();
            document.getElementById('toggle-translation').innerHTML = '🌐 English';
        }
    }

    async translatePage() {
        const elements = document.querySelectorAll('[data-translatable="true"], .post-content p, .post-content h1, .post-content h2, .post-content h3, .post-content li');
        const btn = document.getElementById('toggle-translation');
        btn.disabled = true;
        btn.innerHTML = '⏳ Translating...';

        for (const element of elements) {
            if (element.dataset.originalText) {
                continue; // Already translated
            }

            const text = element.innerText;
            if (text.length > 0 && !this.isEnglish(text)) {
                try {
                    element.dataset.originalText = text;
                    const translatedText = await this.translateText(text, 'ko', 'en');
                    element.innerText = translatedText;
                } catch (error) {
                    console.error('Translation error:', error);
                }
            }
        }

        btn.disabled = false;
        btn.innerHTML = '🇰🇷 한국어';
    }

    restoreOriginal() {
        const elements = document.querySelectorAll('[data-original-text]');
        elements.forEach(element => {
            element.innerText = element.dataset.originalText;
            delete element.dataset.originalText;
        });
    }

    async translateText(text, sourceLang, targetLang) {
        // Use caching to avoid hitting API limits
        const cacheKey = `${sourceLang}-${targetLang}-${text.substring(0, 50)}`;
        
        if (this.translations.has(cacheKey)) {
            return this.translations.get(cacheKey);
        }

        try {
            const response = await fetch(
                `${this.translationApiUrl}?q=${encodeURIComponent(text)}&langpair=${sourceLang}|${targetLang}`,
                { method: 'GET' }
            );

            const data = await response.json();

            if (data.responseStatus === 200) {
                const translatedText = data.responseData.translatedText;
                this.translations.set(cacheKey, translatedText);
                return translatedText;
            } else {
                console.error('Translation API error:', data);
                return text;
            }
        } catch (error) {
            console.error('Translation fetch error:', error);
            return text;
        }
    }

    isEnglish(text) {
        // Check if text contains mostly English characters
        const englishRegex = /[a-zA-Z0-9]/g;
        const matches = text.match(englishRegex) || [];
        return matches.length > text.length * 0.5;
    }
}

// Initialize translator when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.pageTranslator = new PageTranslator();
});

/**
 * Alternative: Using Google Translate API (requires API key)
 * 
 * For production use with more accuracy, consider using Google Cloud Translation API:
 * 
 * async function translateWithGoogle(text, targetLanguage) {
 *     const apiKey = 'YOUR_GOOGLE_API_KEY';
 *     const response = await fetch('https://translation.googleapis.com/language/translate/v2', {
 *         method: 'POST',
 *         body: JSON.stringify({
 *             q: text,
 *             target: targetLanguage,
 *             key: apiKey
 *         }),
 *         headers: {
 *             'Content-Type': 'application/json',
 *         }
 *     });
 *     
 *     const data = await response.json();
 *     return data.data.translations[0].translatedText;
 * }
 */
