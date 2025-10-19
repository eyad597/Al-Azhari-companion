/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI } from '@google/genai';
import { marked } from 'marked';
import * as pdfjsLib from 'pdfjs-dist';
import { PageViewport } from 'pdfjs-dist';

// Configure the PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc =
  'https://esm.sh/pdfjs-dist@4.4.168/build/pdf.worker.mjs';

// --- Web Speech API Type Definitions ---
interface SpeechRecognitionResult {
  readonly isFinal: boolean;
  readonly length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
}
interface SpeechRecognitionAlternative {
  readonly transcript: string;
  readonly confidence: number;
}
interface SpeechRecognitionResultList {
  readonly length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
}
interface SpeechRecognitionEvent extends Event {
  readonly resultIndex: number;
  readonly results: SpeechRecognitionResultList;
}
interface SpeechRecognitionErrorEvent extends Event {
  readonly error: string;
}
interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  lang: string;
  interimResults: boolean;
  onresult: (event: SpeechRecognitionEvent) => void;
  onerror: (event: SpeechRecognitionErrorEvent) => void;
  onstart: () => void;
  onend: () => void;
  start: () => void;
  stop: () => void;
}

// --- Type Definitions ---
type GenerativePart = { inlineData: { data: string; mimeType: string } };
type ConversationTurn = { userPrompt: string; modelResponse: string };
interface ChatSession {
  id: string;
  title: string;
  history: ConversationTurn[];
  pdfFileName: string | null;
  selectedPages: number[];
}


// --- Constants ---
const SESSIONS_KEY = 'gemini-pdf-chat-sessions-v3';
const ACTIVE_SESSION_KEY = 'gemini-pdf-active-session-v3';
const DB_NAME = 'pdf-cache-db-v3';
const STORE_NAME = 'pdf-store';
const THEME_KEY = 'gemini-pdf-chat-theme-v1';
const API_KEY_KEY = 'gemini-pdf-chat-api-key-v1';
const SpeechRecognitionAPI =
  (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

// --- DOM Elements ---
const dom = {
  appLayout: document.querySelector('.app-layout') as HTMLDivElement,
  overlay: document.querySelector('.overlay') as HTMLDivElement,
  form: document.getElementById('prompt-form') as HTMLFormElement,
  fileUpload: document.getElementById('file-upload') as HTMLInputElement,
  dropZone: document.getElementById('drop-zone') as HTMLDivElement,
  fileInfoContainer: document.getElementById('file-info-container') as HTMLDivElement,
  fileName: document.getElementById('file-name') as HTMLSpanElement,
  changeFileButton: document.getElementById('change-file-button') as HTMLButtonElement,
  pageSelectorContainer: document.getElementById('page-selector-container') as HTMLDivElement,
  pageSelectorGrid: document.getElementById('page-selector-grid') as HTMLDivElement,
  promptInput: document.getElementById('prompt-input') as HTMLTextAreaElement,
  micButton: document.getElementById('mic-button') as HTMLButtonElement,
  submitButton: document.getElementById('submit-button') as HTMLButtonElement,
  sendIcon: document.querySelector('#submit-button .send-icon') as SVGElement,
  submitSpinner: document.querySelector('#submit-button .spinner') as HTMLDivElement,
  chatMessages: document.getElementById('chat-messages') as HTMLElement,
  newChatButton: document.getElementById('new-chat-button') as HTMLButtonElement,
  historyList: document.getElementById('history-list') as HTMLElement,
  welcomeMessage: document.getElementById('welcome-message') as HTMLDivElement,
  menuToggleButton: document.getElementById('menu-toggle-button') as HTMLButtonElement,
  themeSwitcher: document.querySelector('.theme-switcher') as HTMLDivElement,
  themeButtons: document.querySelectorAll('.theme-switcher button') as NodeListOf<HTMLButtonElement>,
  // API Key Modal Elements
  settingsButton: document.getElementById('settings-button') as HTMLButtonElement,
  apiKeyModalOverlay: document.getElementById('api-key-modal-overlay') as HTMLDivElement,
  apiKeyModal: document.getElementById('api-key-modal') as HTMLDivElement,
  apiKeyForm: document.getElementById('api-key-form') as HTMLFormElement,
  apiKeyInput: document.getElementById('api-key-input') as HTMLInputElement,
  apiKeyModalClose: document.getElementById('api-key-modal-close') as HTMLButtonElement,
  toggleApiKeyVisibilityButton: document.getElementById('toggle-api-key-visibility') as HTMLButtonElement,
};

// --- App State ---
const state = {
  pdfDocument: null as pdfjsLib.PDFDocumentProxy | null,
  selectedPages: new Set<number>(),
  chatHistory: [] as ConversationTurn[],
  sessions: [] as ChatSession[],
  currentSessionId: null as string | null,
  recognition: null as SpeechRecognition | null,
  isListening: false,
  recognitionErrorOccurred: false,
  synth: window.speechSynthesis,
  voices: [] as SpeechSynthesisVoice[],
  apiKey: null as string | null,
  ai: null as GoogleGenAI | null,
};

// --- Gemini AI Setup ---
const SYSTEM_INSTRUCTION = `أنت معلم خبير ومتخصص في مناهج اللغة العربية للمرحلة الثانوية الأزهرية في مصر. مهمتك هي مساعدة الطلاب على فهم دروسهم. عندما يعطيك الطالب صورة أو عدة صور من كتاب ويطرح سؤالاً، قم بتحليل النصوص في جميع الصور أولاً، ثم أجب على سؤاله بإجابة شاملة ولكن متوسطة الطول وموجزة. ركز على النقاط الأساسية وقدم شرحًا واضحًا ومباشرًا. استخدم التنسيق مثل القوائم والنقاط والعناوين لجعل إجابتك سهلة القراءة والفهم. كن دقيقاً في معلوماتك واعتمد على القواعد النحوية والبلاغية المقررة في المنهج الأزهري. ملاحظة هامة جداً: أنت ممنوع منعاً باتاً من استخدام تنسيق LaTeX الرياضي أو أي رموز برمجية أخرى. جميع إجاباتك يجب أن تكون نصاً عادياً بسيطاً ومفهوماً تماماً. إذا احتجت إلى كتابة معادلات أو رموز، فاكتبها كنص عادي تمامًا كما تظهر في الكتب المدرسية.`;


// --- IndexedDB Helper ---
const db = {
  open(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, 1);
      request.onerror = () => reject('Error opening IndexedDB');
      request.onsuccess = () => resolve(request.result);
      request.onupgradeneeded = (event) => {
        const dbInstance = (event.target as IDBOpenDBRequest).result;
        if (!dbInstance.objectStoreNames.contains(STORE_NAME)) {
          dbInstance.createObjectStore(STORE_NAME);
        }
      };
    });
  },
  async set(key: string, value: ArrayBuffer) {
    const dbInstance = await this.open();
    return new Promise<void>((resolve, reject) => {
      const transaction = dbInstance.transaction(STORE_NAME, 'readwrite');
      transaction.oncomplete = () => { dbInstance.close(); resolve(); };
      transaction.onerror = () => { dbInstance.close(); reject('Transaction error'); };
      transaction.objectStore(STORE_NAME).put(value, key);
    });
  },
  async get(key: string): Promise<ArrayBuffer | null> {
    const dbInstance = await this.open();
    return new Promise((resolve, reject) => {
      const transaction = dbInstance.transaction(STORE_NAME, 'readonly');
      transaction.oncomplete = () => dbInstance.close();
      transaction.onerror = () => { dbInstance.close(); reject('Transaction error'); };
      const request = transaction.objectStore(STORE_NAME).get(key);
      request.onsuccess = () => resolve(request.result || null);
    });
  },
  async del(key: string) {
    const dbInstance = await this.open();
    return new Promise<void>((resolve, reject) => {
      const transaction = dbInstance.transaction(STORE_NAME, 'readwrite');
      transaction.oncomplete = () => { dbInstance.close(); resolve(); };
      transaction.onerror = () => { dbInstance.close(); reject('Transaction error'); };
      transaction.objectStore(STORE_NAME).delete(key);
    });
  }
};


// --- API Key Management ---
function initializeAi(key: string): boolean {
    if (!key) {
      console.error("API key is missing.");
      return false;
    }
    try {
      state.ai = new GoogleGenAI({ apiKey: key });
      state.apiKey = key;
      console.log("Gemini AI initialized successfully.");
      return true;
    } catch (e) {
      console.error("Failed to initialize GoogleGenAI:", e);
      const errorMessage = e instanceof Error ? e.message : String(e);
      alert(`فشل تهيئة واجهة برمجة التطبيقات. قد يكون المفتاح غير صالح. الخطأ: ${errorMessage}`);
      state.ai = null;
      state.apiKey = null;
      return false;
    }
}

function showApiKeyModal() {
    dom.apiKeyModalOverlay.classList.remove('hidden');
    document.body.classList.add('modal-open');
    dom.apiKeyInput.value = state.apiKey || '';
    dom.apiKeyModalClose.classList.toggle('hidden', !state.apiKey);
}

function hideApiKeyModal() {
    dom.apiKeyModalOverlay.classList.add('hidden');
    document.body.classList.remove('modal-open');
}

async function handleApiKeyFormSubmit(e: Event) {
    e.preventDefault();
    const newKey = dom.apiKeyInput.value.trim();
    if (!newKey) {
        alert("الرجاء إدخال مفتاح API.");
        return;
    }
    localStorage.setItem(API_KEY_KEY, newKey);
    if (initializeAi(newKey)) {
        hideApiKeyModal();
    }
}


// --- UI Functions ---
const ui = {
  setLoading(isLoading: boolean) {
    dom.submitButton.disabled = isLoading;
    dom.sendIcon.classList.toggle('hidden', isLoading);
    dom.submitSpinner.classList.toggle('hidden', !isLoading);
  },
  
  enablePrompt(enabled: boolean) {
    dom.promptInput.disabled = !enabled;
    dom.micButton.disabled = !enabled;
    dom.submitButton.disabled = !enabled;
    if (!enabled) {
      dom.promptInput.value = '';
      dom.promptInput.placeholder = 'الرجاء تحديد صفحة واحدة على الأقل...';
    } else {
      dom.promptInput.placeholder = 'اكتب سؤالك هنا...';
    }
  },

  resetFileInput() {
    dom.pageSelectorContainer.classList.add('hidden');
    dom.fileInfoContainer.classList.add('hidden');
    dom.pageSelectorGrid.innerHTML = '';
    dom.dropZone.classList.remove('hidden');
    dom.fileUpload.value = '';
    this.enablePrompt(false);
  },
  
  createMessageElement(role: 'user' | 'model', content: string): HTMLElement {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}`;
    const avatar = role === 'user'
      ? `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`
      : `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path></svg>`;
    
    messageDiv.innerHTML = `
      <div class="avatar">${avatar}</div>
      <div class="message-content">${content}</div>
    `;
    return messageDiv;
  },

  showTypingIndicator(): HTMLElement {
    const indicator = this.createMessageElement('model', `
      <div class="typing-indicator">
        <span></span><span></span><span></span>
      </div>
    `);
    dom.chatMessages.appendChild(indicator);
    scrollToBottom();
    return indicator;
  },
  
  showProgressIndicator(text: string): HTMLElement {
    const indicator = document.createElement('div');
    indicator.className = 'progress-indicator';
    indicator.innerHTML = `
        <div class="spinner"></div>
        <p>${text}</p>
    `;
    dom.chatMessages.appendChild(indicator);
    scrollToBottom();
    return indicator;
  },

  async renderChatHistory() {
    speech.stop();
    dom.chatMessages.innerHTML = '';
    
    dom.chatMessages.appendChild(dom.welcomeMessage);

    if (state.chatHistory.length === 0) {
      dom.welcomeMessage.style.display = 'flex';
      return;
    }
    dom.welcomeMessage.style.display = 'none';

    for (const turn of state.chatHistory) {
      const userMsg = this.createMessageElement('user', `<p>${turn.userPrompt}</p>`);
      dom.chatMessages.appendChild(userMsg);

      const parsedHtml = await marked.parse(turn.modelResponse);
      const modelMsg = this.createMessageElement('model', parsedHtml as string);
      
      const actions = document.createElement('div');
      actions.className = 'model-response-actions';
      
      const speakerButton = speech.createSpeakerButton(turn.modelResponse);
      const copyButton = createCopyButton(turn.modelResponse);
      
      actions.appendChild(speakerButton);
      actions.appendChild(copyButton);

      modelMsg.querySelector('.message-content')?.appendChild(actions);

      dom.chatMessages.appendChild(modelMsg);
    }
    scrollToBottom();
  },
  
  renderHistoryList() {
      dom.historyList.innerHTML = '';
      if (state.sessions.length === 0) {
          dom.historyList.innerHTML = `<p class="no-history">لا توجد محادثات سابقة.</p>`;
          return;
      }
      for (const session of state.sessions) {
          const item = document.createElement('div');
          item.className = 'history-item';
          item.dataset.sessionId = session.id;
          if (session.id === state.currentSessionId) {
              item.classList.add('active');
          }
          
          item.innerHTML = `
            <div class="history-item-main">
                <svg class="history-item-icon" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
                <span class="history-item-title">${session.title}</span>
            </div>
            <div class="history-item-actions">
                <button class="delete-chat-button" title="حذف المحادثة">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                </button>
            </div>
          `;

          item.addEventListener('click', (e) => {
            if ((e.target as HTMLElement).closest('.delete-chat-button')) return;
            loadSession(session.id);
          });
          
          item.querySelector('.delete-chat-button')?.addEventListener('click', (e) => {
              e.stopPropagation();
              if (confirm(`هل أنت متأكد من حذف محادثة "${session.title}"؟`)) {
                  deleteSession(session.id);
              }
          });

          dom.historyList.appendChild(item);
      }
  }
};


// --- Theme Management ---
function applyTheme(theme: 'light' | 'dark' | 'system') {
    if (theme === 'system') {
        document.documentElement.removeAttribute('data-theme');
        localStorage.removeItem(THEME_KEY);
    } else {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem(THEME_KEY, theme);
    }

    dom.themeButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.theme === theme);
    });
}

function initTheme() {
    const savedTheme = localStorage.getItem(THEME_KEY) as 'light' | 'dark' | 'system' | null;
    const theme = savedTheme || 'system';
    applyTheme(theme);

    dom.themeSwitcher.addEventListener('click', (e) => {
        const target = e.target as HTMLElement;
        const button = target.closest('button');
        if (button && button.dataset.theme) {
            applyTheme(button.dataset.theme as 'light' | 'dark' | 'system');
        }
    });
}


// --- Speech Synthesis & Recognition ---
let speechIsManuallyStopped = false;

const speech = {
  populateVoiceList() {
    state.voices = state.synth.getVoices().filter(voice => voice.lang.startsWith('ar'));
  },

  stop() {
    speechIsManuallyStopped = true;
    if (state.synth.speaking) {
      state.synth.cancel();
    }
    // This is the key change for UI reliability.
    // Instead of relying on a fragile event chain to clean up the button,
    // we do it here, directly and synchronously. This prevents the button
    // from getting "stuck" in a speaking state.
    const speakingButton = document.querySelector('.speaker-button.speaking') as HTMLButtonElement;
    if (speakingButton) {
        speakingButton.classList.remove('speaking');
        speakingButton.innerHTML = speakingButton.dataset.originalContent || '<span>قراءة</span>';
    }
  },

  createSpeakerButton(textToRead: string): HTMLButtonElement {
    const button = document.createElement('button');
    button.className = 'speaker-button';
    button.title = 'قراءة الإجابة صوتيًا';
    const originalContent = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg><span>قراءة</span>`;
    const speakingContent = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="6" width="12" height="12"></rect></svg><span>إيقاف</span>`;
    button.innerHTML = originalContent;
    button.dataset.originalContent = originalContent;

    const cleanup = () => {
        button.classList.remove('speaking');
        button.innerHTML = originalContent;
    };

    button.addEventListener('click', () => {
      if (button.classList.contains('speaking')) {
        speech.stop();
        return;
      }
      
      speech.stop(); // Stop any other speech (this will also reset its button UI).
      speechIsManuallyStopped = false; // Reset flag for this new playback.

      // Split text into manageable chunks to avoid 'synthesis-failed' on long texts.
      const MAX_CHUNK_LENGTH = 180; 
      const sentences = textToRead.match(/[^.!?؟\n\r]+[.!?؟\n\r]?/g) || [];
      const chunks: string[] = [];
      
      sentences.forEach(sentence => {
          let cleanSentence = sentence.trim();
          if (cleanSentence.length > MAX_CHUNK_LENGTH) {
              // Further split very long sentences
              let currentPart = cleanSentence;
              while (currentPart.length > 0) {
                  chunks.push(currentPart.substring(0, MAX_CHUNK_LENGTH));
                  currentPart = currentPart.substring(MAX_CHUNK_LENGTH);
              }
          } else if (cleanSentence) {
              chunks.push(cleanSentence);
          }
      });

      if (chunks.length === 0) return;

      button.classList.add('speaking');
      button.innerHTML = speakingContent;
      
      const speakChunk = (index: number) => {
          if (index >= chunks.length || speechIsManuallyStopped) {
              // If stopped manually, the UI was handled by speech.stop().
              // If finished naturally, call cleanup.
              if (!speechIsManuallyStopped) {
                  cleanup();
              }
              return;
          }

          const chunk = chunks[index];
          const utterance = new SpeechSynthesisUtterance(chunk);

          utterance.lang = 'ar-SA';
          if (state.voices.length > 0) {
            utterance.voice = state.voices[0];
          }
          
          utterance.onend = () => {
            // Short delay between chunks for more natural pacing
            setTimeout(() => speakChunk(index + 1), 100); 
          };

          utterance.onerror = (event) => {
              console.error('SpeechSynthesisUtterance error:', (event as SpeechSynthesisErrorEvent).error, "on chunk:", chunk);
              // On error, call the main stop function to ensure a full, clean reset.
              speech.stop();
          };
          
          state.synth.speak(utterance);
      };
      
      // Defer the initial speak call to prevent race conditions with `cancel()`.
      setTimeout(() => {
        // Re-check the flag in case the user clicked stop during the timeout.
        if (!speechIsManuallyStopped) {
          speakChunk(0);
        }
      }, 50);
    });
    return button;
  },
  
  toggleListening() {
    if (!state.recognition) return;
    if (state.isListening) {
      state.recognition.stop();
    } else {
      speech.stop();
      dom.promptInput.value = '';
      dom.promptInput.placeholder = '... استمع الآن';
      state.recognition.start();
    }
  },

  setupRecognition() {
    if (!SpeechRecognitionAPI) {
      console.warn('Speech Recognition API not supported.');
      dom.micButton.style.display = 'none';
      return;
    }
    state.recognition = new SpeechRecognitionAPI();
    state.recognition.continuous = false;
    state.recognition.lang = 'ar-SA';
    state.recognition.interimResults = false;

    state.recognition.onstart = () => {
      state.isListening = true;
      dom.micButton.classList.add('recording');
      state.recognitionErrorOccurred = false;
    };
    state.recognition.onend = () => {
      state.isListening = false;
      dom.micButton.classList.remove('recording');
      if (!state.recognitionErrorOccurred) {
        ui.enablePrompt(state.selectedPages.size > 0);
      }
    };
    state.recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      state.recognitionErrorOccurred = true;
      console.error('Speech recognition error:', event.error);
      const errorPlaceholders: { [key: string]: string } = {
        'no-speech': 'لم يتم سماع أي صوت. يرجى المحاولة مرة أخرى.',
        'audio-capture': 'مشكلة في الميكروفون. يرجى التحقق من الإعدادات.',
        'not-allowed': 'تم رفض الوصول إلى الميكروفون.',
      };
      dom.promptInput.placeholder = errorPlaceholders[event.error] || 'حدث خطأ في التعرف على الصوت.';
    };
    state.recognition.onresult = (event: SpeechRecognitionEvent) => {
      dom.promptInput.value = event.results[0][0].transcript;
      autoResizeTextarea();
    };
  }
};


// --- Helper Functions ---
function scrollToBottom() {
    // Defer the scroll action until after the browser has finished rendering
    // the new content and updated the layout. Using setTimeout with a 0ms delay
    // pushes this task to the end of the event queue, which is a reliable way
    // to ensure the scrollHeight property is up-to-date.
    setTimeout(() => {
        dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    }, 0);
}

function createCopyButton(textToCopy: string): HTMLButtonElement {
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.title = 'نسخ الإجابة';
    const originalContent = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg><span>نسخ</span>`;
    const copiedContent = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg><span>تم النسخ</span>`;
    button.innerHTML = originalContent;

    button.addEventListener('click', () => {
        navigator.clipboard.writeText(textToCopy).then(() => {
            button.innerHTML = copiedContent;
            button.classList.add('copied');
            setTimeout(() => {
                button.innerHTML = originalContent;
                button.classList.remove('copied');
            }, 2000);
        });
    });
    return button;
}

function autoResizeTextarea() {
    dom.promptInput.style.height = 'auto';
    dom.promptInput.style.height = `${dom.promptInput.scrollHeight}px`;
}


// --- PDF & Core Logic ---
async function getSelectedPageImages(
  onProgress: (processed: number, total: number) => void
): Promise<GenerativePart[]> {
  if (!state.pdfDocument) return [];

  const allParts: { pageNum: number, part: GenerativePart }[] = [];
  const sortedPages = Array.from(state.selectedPages).sort((a, b) => a - b);

  let processedCount = 0;
  onProgress(processedCount, sortedPages.length);
  
  const CONCURRENCY = 4;
  for (let i = 0; i < sortedPages.length; i += CONCURRENCY) {
    const pageChunk = sortedPages.slice(i, i + CONCURRENCY);
    const chunkPromises = pageChunk.map(async (pageNum) => {
      try {
        const page = await state.pdfDocument.getPage(pageNum);
        // Performance Tweak: reduced scale and quality
        const viewport: PageViewport = page.getViewport({ scale: 0.9 });
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.height = viewport.height;
        canvas.width = viewport.width;

        if (context) {
          // FIX: Added 'canvas' property to the render parameters. The TypeScript error suggests
          // a type definition is being used that requires this property.
          await page.render({ canvasContext: context, viewport, canvas }).promise;
          const base64Data = canvas.toDataURL('image/jpeg', 0.75).split(',')[1];
          return { pageNum, part: { inlineData: { data: base64Data, mimeType: 'image/jpeg' } } };
        }
      } catch (e) {
        console.error(`Failed to process page ${pageNum}:`, e);
      }
      return null;
    });

    const processedChunks = await Promise.all(chunkPromises);
    
    for (const result of processedChunks) {
      if (result) {
        allParts.push(result);
      }
      processedCount++;
      onProgress(processedCount, sortedPages.length);
    }
    await new Promise(resolve => setTimeout(resolve, 0));
  }

  return allParts
    .sort((a, b) => a.pageNum - b.pageNum)
    .map(item => item.part);
}

async function renderPagePreviews() {
  if (!state.pdfDocument) return;
  dom.pageSelectorGrid.innerHTML = '';
  for (let i = 1; i <= state.pdfDocument.numPages; i++) {
    const page = await state.pdfDocument.getPage(i);
    const viewport = page.getViewport({ scale: 0.3 });
    const card = document.createElement('div');
    card.className = 'page-preview-card';
    card.dataset.pageNum = String(i);
    if (state.selectedPages.has(i)) card.classList.add('selected');

    const canvas = document.createElement('canvas');
    canvas.height = viewport.height;
    canvas.width = viewport.width;
    card.appendChild(canvas);

    const overlay = document.createElement('div');
    overlay.className = 'page-preview-overlay';
    overlay.innerHTML = `<span class="page-number-display">صفحة ${i}</span>`;
    card.appendChild(overlay);

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.className = 'page-preview-checkbox';
    checkbox.checked = state.selectedPages.has(i);
    card.appendChild(checkbox);
    
    dom.pageSelectorGrid.appendChild(card);
    // FIX: Added 'canvas' property to render parameters and refactored to avoid calling getContext twice.
    const context = canvas.getContext('2d');
    if (context) {
      page.render({ canvasContext: context, viewport, canvas });
    }
  }
}

async function processPdfFile(fileBuffer: ArrayBuffer, fileName: string) {
    const loadingTask = pdfjsLib.getDocument(fileBuffer);
    try {
        state.pdfDocument = await loadingTask.promise;
        dom.fileName.textContent = fileName;
        dom.dropZone.classList.add('hidden');
        dom.fileInfoContainer.classList.remove('hidden');
        dom.pageSelectorContainer.classList.remove('hidden');
        await renderPagePreviews();
        ui.enablePrompt(state.selectedPages.size > 0);
    } catch (error) {
        console.error("Failed to process PDF:", error);
        alert("حدث خطأ أثناء معالجة ملف PDF. يرجى التأكد من أن الملف غير تالف.");
        startNewSession(); 
    }
}


// --- History & Session Management ---
function saveSessionsToStorage() {
  localStorage.setItem(SESSIONS_KEY, JSON.stringify(state.sessions));
  if(state.currentSessionId) {
    localStorage.setItem(ACTIVE_SESSION_KEY, state.currentSessionId);
  }
}

function loadSessionsFromStorage() {
  const saved = localStorage.getItem(SESSIONS_KEY);
  if (saved) {
    state.sessions = JSON.parse(saved);
  }
}

async function startNewSession() {
    speech.stop();
    
    const newId = Date.now().toString();
    const newSession: ChatSession = {
        id: newId,
        title: 'محادثة جديدة',
        history: [],
        pdfFileName: null,
        selectedPages: [],
    };
    
    state.sessions.unshift(newSession);
    saveSessionsToStorage();
    await loadSession(newId);
}

async function deleteSession(sessionId: string) {
    state.sessions = state.sessions.filter(s => s.id !== sessionId);
    await db.del(`pdf-${sessionId}`);
    saveSessionsToStorage();
    
    if (state.currentSessionId === sessionId) {
        if (state.sessions.length > 0) {
            await loadSession(state.sessions[0].id);
        } else {
            await startNewSession();
        }
    } else {
        ui.renderHistoryList();
    }
}

async function loadSession(sessionId: string) {
    speech.stop();
    ui.resetFileInput();

    const session = state.sessions.find(s => s.id === sessionId);
    if (!session) {
        console.error("Session not found:", sessionId);
        if (state.sessions.length > 0) {
            await loadSession(state.sessions[0].id);
        } else {
            await startNewSession();
        }
        return;
    }

    state.currentSessionId = sessionId;
    state.chatHistory = session.history;
    state.selectedPages = new Set(session.selectedPages);
    state.pdfDocument = null;

    await ui.renderChatHistory();
    ui.renderHistoryList();
    saveSessionsToStorage();

    if (session.pdfFileName) {
        const pdfBuffer = await db.get(`pdf-${session.id}`);
        if (pdfBuffer) {
            await processPdfFile(pdfBuffer, session.pdfFileName);
        } else {
            console.warn(`PDF for session ${sessionId} not found in DB.`);
        }
    } else {
        ui.enablePrompt(false);
        dom.promptInput.placeholder = 'الرجاء رفع ملف PDF لبدء المحادثة.';
    }
}


// --- Event Handlers ---
async function handleFormSubmit(e: Event) {
  e.preventDefault();
  speech.stop();

  if (!state.ai) {
    alert("الرجاء إعداد مفتاح API أولاً.");
    showApiKeyModal();
    return;
  }

  const promptText = dom.promptInput.value.trim();
  if (!promptText || state.selectedPages.size === 0) {
    alert('الرجاء تحديد صفحة واحدة على الأقل وكتابة سؤال.');
    return;
  }
  
  if (state.selectedPages.size > 30) {
    alert('لضمان أفضل أداء، يمكنك تحليل 30 صفحة كحد أقصى في كل مرة. يرجى تقليل عدد الصفحات المحددة.');
    return;
  }

  ui.setLoading(true);
  dom.welcomeMessage.style.display = 'none';
  const userMsg = ui.createMessageElement('user', `<p>${promptText}</p>`);
  dom.chatMessages.appendChild(userMsg);
  dom.promptInput.value = '';
  autoResizeTextarea();
  scrollToBottom();
  
  const progressIndicator = ui.showProgressIndicator('جارِ تحليل الصفحات...');

  try {
    const imageParts = await getSelectedPageImages((processed, total) => {
        progressIndicator.querySelector('p')!.textContent = `جارِ تحليل الصفحات... (${processed}/${total})`;
    });
    
    progressIndicator.remove();

    if (imageParts.length === 0 && state.selectedPages.size > 0) {
      throw new Error("لم يتم استخراج صور من الصفحات المحددة.");
    }
    
    const typingIndicator = ui.showTypingIndicator();
    const modelContent = typingIndicator.querySelector('.message-content')!;
    
    const stream = await state.ai.models.generateContentStream({
      model: 'gemini-2.5-flash',
      contents: [{ parts: [...imageParts, { text: promptText }] }],
      config: { 
        systemInstruction: SYSTEM_INSTRUCTION,
      },
    });
    
    let modelResponseText = '';
    for await (const chunk of stream) {
        modelResponseText += chunk.text;
        // Using `as string` because marked can return a promise
        modelContent.innerHTML = await marked.parse(modelResponseText) as string;
        scrollToBottom();
    }

    if (!modelResponseText || modelResponseText.trim() === '') {
        throw new Error("ورد رد فارغ من الذكاء الاصطناعي. حاول مرة أخرى.");
    }
    
    const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
    if(currentSession) {
        if (currentSession.history.length === 0) {
            currentSession.title = promptText.substring(0, 40) + (promptText.length > 40 ? '...' : '');
            ui.renderHistoryList();
        }
        currentSession.history.push({ userPrompt: promptText, modelResponse: modelResponseText });
        state.chatHistory = currentSession.history;
        saveSessionsToStorage();
    }
    
    const actions = document.createElement('div');
    actions.className = 'model-response-actions';
    actions.appendChild(speech.createSpeakerButton(modelResponseText));
    actions.appendChild(createCopyButton(modelResponseText));
    modelContent.appendChild(actions);

  } catch (error) {
    if(progressIndicator) progressIndicator.remove();
    const existingTypingIndicator = dom.chatMessages.querySelector('.typing-indicator')?.closest('.chat-message');
    
    let errorMessage = "حدث خطأ غير متوقع. يرجى المحاولة مرة أخرى.";
    if (error instanceof Error) {
        // Check for specific authentication/permission errors (like 403)
        if (error.message.includes('403') || /API[-_ ]?key/i.test(error.message)) {
            errorMessage = `حدث خطأ في المصادقة. قد يكون هذا بسبب عدم صلاحية مفتاح API أو عدم تمكين الفوترة لمشروعك. يرجى التحقق من مفتاح API الخاص بك بالضغط على أيقونة الإعدادات.`;
        } else {
            errorMessage = error.message;
        }
    }

    const errorContent = `<p><strong>حدث خطأ:</strong> ${errorMessage}</p>`;
    const errorHtml = `<div class="gemini-error">${errorContent}</div>`;

    if (existingTypingIndicator) {
        existingTypingIndicator.querySelector('.message-content')!.innerHTML = errorHtml;
    } else {
        const errorMsg = ui.createMessageElement('model', errorHtml);
        dom.chatMessages.appendChild(errorMsg);
    }
  } finally {
    ui.setLoading(false);
    scrollToBottom();
  }
}

function handlePageSelection(e: Event) {
  const target = e.target as HTMLElement;
  const card = target.closest('.page-preview-card') as HTMLDivElement;
  if (!card) return;

  const pageNum = parseInt(card.dataset.pageNum!, 10);
  const checkbox = card.querySelector('.page-preview-checkbox') as HTMLInputElement;

  if (target.tagName !== 'INPUT') checkbox.checked = !checkbox.checked;
  
  if (checkbox.checked) {
    state.selectedPages.add(pageNum);
    card.classList.add('selected');
  } else {
    state.selectedPages.delete(pageNum);
    card.classList.remove('selected');
  }
  
  const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
  if (currentSession) {
      currentSession.selectedPages = Array.from(state.selectedPages);
      saveSessionsToStorage();
  }
  ui.enablePrompt(state.selectedPages.size > 0);
}

async function handleFile(file: File) {
  if (!file || file.type !== 'application/pdf') {
    alert('Please select a valid PDF file.');
    return;
  }
  await startNewSession();

  const fileBuffer = await file.arrayBuffer();
  const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
  if (currentSession) {
      currentSession.pdfFileName = file.name;
      await db.set(`pdf-${currentSession.id}`, fileBuffer);
      saveSessionsToStorage();
      await processPdfFile(fileBuffer, file.name);
  }
}


// --- App Initialization ---
async function init() {
  // Event listeners
  dom.form.addEventListener('submit', handleFormSubmit);
  dom.micButton.addEventListener('click', speech.toggleListening);
  dom.newChatButton.addEventListener('click', startNewSession);
  dom.changeFileButton.addEventListener('click', () => dom.fileUpload.click());
  dom.pageSelectorGrid.addEventListener('click', handlePageSelection);
  dom.fileUpload.addEventListener('change', (e) => {
    const files = (e.target as HTMLInputElement).files;
    if (files?.[0]) handleFile(files[0]);
  });
  dom.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dom.dropZone.classList.add('drag-over');
  });
  dom.dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dom.dropZone.classList.remove('drag-over');
  });
  dom.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dom.dropZone.classList.remove('drag-over');
    if (e.dataTransfer?.files?.[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });
  dom.promptInput.addEventListener('input', autoResizeTextarea);

  dom.menuToggleButton.addEventListener('click', () => {
    dom.appLayout.classList.add('history-open');
    dom.overlay.classList.remove('hidden');
  });
  dom.overlay.addEventListener('click', () => {
    dom.appLayout.classList.remove('history-open');
    dom.overlay.classList.add('hidden');
  });
  dom.historyList.addEventListener('click', (e) => {
    // Hide sidebar on mobile after clicking an item
    if (window.innerWidth <= 768 && (e.target as HTMLElement).closest('.history-item')) {
      dom.appLayout.classList.remove('history-open');
      dom.overlay.classList.add('hidden');
    }
  });

  // API Key Modal Listeners
  dom.settingsButton.addEventListener('click', showApiKeyModal);
  dom.apiKeyForm.addEventListener('submit', handleApiKeyFormSubmit);
  dom.apiKeyModalClose.addEventListener('click', hideApiKeyModal);
  dom.toggleApiKeyVisibilityButton.addEventListener('click', () => {
      const isPassword = dom.apiKeyInput.type === 'password';
      dom.apiKeyInput.type = isPassword ? 'text' : 'password';
      dom.toggleApiKeyVisibilityButton.querySelector('.eye-icon')?.classList.toggle('hidden', isPassword);
      dom.toggleApiKeyVisibilityButton.querySelector('.eye-off-icon')?.classList.toggle('hidden', !isPassword);
  });

  // Initial setup
  initTheme();
  speech.populateVoiceList();
  if (speechSynthesis.onvoiceschanged !== undefined) {
    speechSynthesis.onvoiceschanged = speech.populateVoiceList;
  }
  speech.setupRecognition();
  
  // API Key initialization
  const savedApiKey = localStorage.getItem(API_KEY_KEY);
  if (savedApiKey) {
    initializeAi(savedApiKey);
  } else {
    showApiKeyModal();
  }
  
  loadSessionsFromStorage();
  const lastActiveId = localStorage.getItem(ACTIVE_SESSION_KEY);
  
  if (lastActiveId && state.sessions.some(s => s.id === lastActiveId)) {
      await loadSession(lastActiveId);
  } else if (state.sessions.length > 0) {
      await loadSession(state.sessions[0].id);
  } else {
      await startNewSession();
  }
}

init();