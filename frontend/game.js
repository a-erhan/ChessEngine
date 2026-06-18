/* ────────────────────────────────────────────────────────────────────
   game.js — Chess Engine AI Frontend Logic
   ──────────────────────────────────────────────────────────────────── */

'use strict';

// ══════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ══════════════════════════════════════════════════════════════════════

// Backend URL otomatik algılama:
// - Lokal geliştirme: http://localhost:5001
// - Production (GitHub Pages): window.CHESS_API_URL'den okur
//   (config.js içinde tanımlanır, deploy sonrası güncellenir)
const isLocalhost = ['localhost', '127.0.0.1', ''].includes(window.location.hostname);
const API_BASE = window.CHESS_API_URL
  || (isLocalhost ? 'http://localhost:5001' : 'RENDER_URL_BURAYA');

const MODEL_INFO = {
  agresif:  { emoji: '🔥', label: 'Agresif',  tag: 'Phase1 E10' },
  tedbirli: { emoji: '🛡️', label: 'Tedbirli', tag: 'Finetune E5' },
  akil:     { emoji: '🧠', label: 'Akıl',     tag: 'Ultra E3' }
};

// ══════════════════════════════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════════════════════════════

let game = null;            // chess.js instance
let board = null;           // chessboard.js instance
let gameState = {
  mode: 'human-white',      // 'human-white' | 'human-black' | 'bot-vs-bot'
  model1: 'agresif',        // beyaz / tek model
  model2: 'tedbirli',       // siyah (bot vs bot'ta kullanılır)
  running: false,
  paused: false,
  botVsBotInterval: null,
  moveDelay: 1000,
  moveHistory: [],           // [{san, uci, fen}]
  lastFrom: null,
  lastTo: null,
};

// ══════════════════════════════════════════════════════════════════════
// UTILS
// ══════════════════════════════════════════════════════════════════════

function showToast(msg, type = 'info', duration = 3500) {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    toast.style.transition = '0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

function setStatus(text, online = null) {
  document.getElementById('statusText').textContent = text;
  const dot = document.getElementById('statusDot');
  if (online === true)  { dot.className = 'status-dot online'; }
  if (online === false) { dot.className = 'status-dot error'; }
  if (online === null)  { dot.className = 'status-dot'; }
}

function setThinking(side, thinking) {
  // side: 'top' | 'bottom'
  const el = document.getElementById(`${side}Thinking`);
  const card = document.getElementById(`${side}PlayerCard`);
  if (!el || !card) return;
  el.style.display = thinking ? 'flex' : 'none';
  if (thinking) card.classList.add('active');
  else          card.classList.remove('active');
}

function updateEvalBar(value) {
  // value: -1 to +1, +1 beyaz kazanıyor
  const pct = Math.round(((value + 1) / 2) * 100);
  const clamped = Math.max(5, Math.min(95, pct));
  document.getElementById('evalFillWhite').style.height = clamped + '%';
  const whiteAdv = value > 0 ? `+${(value * 5).toFixed(1)}` : (value * 5).toFixed(1);
  document.getElementById('evalWhiteLabel').textContent = value > 0 ? whiteAdv : '+0.0';
  document.getElementById('evalBlackLabel').textContent = value < 0 ? `+${(-value * 5).toFixed(1)}` : '+0.0';
}

function addMoveToHistory(moveNum, san, color) {
  const list = document.getElementById('moveList');
  const empty = list.querySelector('.move-list-empty');
  if (empty) empty.remove();

  const rowId = `move-row-${Math.ceil(moveNum / 2)}`;
  let row = document.getElementById(rowId);

  if (!row || color === 'w') {
    row = document.createElement('div');
    row.className = 'move-row';
    row.id = rowId;
    const num = document.createElement('span');
    num.className = 'move-num';
    num.textContent = Math.ceil(moveNum / 2) + '.';
    row.appendChild(num);
    list.appendChild(row);
  }

  const cell = document.createElement('span');
  cell.className = 'move-cell';
  cell.textContent = san;
  row.appendChild(cell);
  list.scrollTop = list.scrollHeight;
  return cell;
}

function highlightLastMove(fromSq, toSq) {
  // Önceki highlight'ları temizle
  document.querySelectorAll('.highlight-from, .highlight-to').forEach(el => {
    el.classList.remove('highlight-from', 'highlight-to');
  });

  if (!fromSq || !toSq) return;
  // chessboard.js square sınıfı: "square-e2" gibi
  const fromEl = document.querySelector(`.square-${fromSq}`);
  const toEl   = document.querySelector(`.square-${toSq}`);
  if (fromEl) fromEl.classList.add('highlight-from');
  if (toEl)   toEl.classList.add('highlight-to');
}

function updateGameInfo(statusText, moveCount) {
  document.getElementById('gameStatusText').textContent = statusText;
  document.getElementById('moveCountText').textContent  = moveCount;
}

// ══════════════════════════════════════════════════════════════════════
// PARTICLE SYSTEM
// ══════════════════════════════════════════════════════════════════════

function initParticles() {
  const container = document.getElementById('particles');
  for (let i = 0; i < 25; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    p.style.left      = Math.random() * 100 + 'vw';
    p.style.animationDuration  = (8 + Math.random() * 12) + 's';
    p.style.animationDelay     = (-Math.random() * 15) + 's';
    p.style.width  = (1 + Math.random() * 2) + 'px';
    p.style.height = p.style.width;
    p.style.opacity = Math.random() * 0.6;
    container.appendChild(p);
  }
}

// ══════════════════════════════════════════════════════════════════════
// API
// ══════════════════════════════════════════════════════════════════════

async function checkBackend() {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      setStatus('Backend bağlı ✓', true);
      showToast('♟ Chess Engine API bağlandı!', 'success');
      return true;
    }
  } catch {
    setStatus('Backend bağlanamadı', false);
    showToast('⚠ Backend çevrimdışı — localhost:5000 başlatın', 'error', 6000);
    return false;
  }
}

async function fetchMove(fen, modelKey) {
  const res = await fetch(`${API_BASE}/api/move`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fen, model: modelKey }),
    signal: AbortSignal.timeout(30000)
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ══════════════════════════════════════════════════════════════════════
// SETUP SCREEN
// ══════════════════════════════════════════════════════════════════════

function initSetupScreen() {
  // Mod seçimi
  document.querySelectorAll('.mode-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.mode-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      gameState.mode = card.dataset.mode;
      updateSetupUI();
    });
  });

  // Model 1 seçimi (slot=1)
  document.querySelectorAll('[data-slot="1"]').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('[data-slot="1"]').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      gameState.model1 = card.dataset.model;
    });
  });

  // Model 2 seçimi (slot=2, bot vs bot)
  document.querySelectorAll('[data-slot="2"]').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('[data-slot="2"]').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      gameState.model2 = card.dataset.model;
    });
  });

  // Başlat
  document.getElementById('startBtn').addEventListener('click', startGame);
}

function updateSetupUI() {
  const isBvB = gameState.mode === 'bot-vs-bot';
  document.getElementById('bot2Section').style.display = isBvB ? 'block' : 'none';

  const labels = {
    'human-white': 'AI Modeli Seç (Siyah Oynar)',
    'human-black': 'AI Modeli Seç (Beyaz Oynar)',
    'bot-vs-bot':  'Beyaz Taraf (1. Model)',
  };
  document.getElementById('modelSectionLabel').textContent = labels[gameState.mode];

  // Beyaz/Siyah badge'lerini güncelle
  document.querySelectorAll('[data-slot="1"] .model-badge').forEach(b => {
    b.textContent = isBvB ? 'Beyaz' : (gameState.mode === 'human-black' ? 'Beyaz AI' : 'Siyah AI');
    b.style.display = 'block';
  });
}

// ══════════════════════════════════════════════════════════════════════
// GAME INITIALIZATION
// ══════════════════════════════════════════════════════════════════════

function startGame() {
  document.getElementById('setupScreen').classList.add('hidden');
  document.getElementById('gameScreen').classList.remove('hidden');
  initGame();
}

function initGame() {
  // Reset state
  game = new Chess();
  gameState.running   = true;
  gameState.paused    = false;
  gameState.moveHistory = [];

  // Move list temizle
  const list = document.getElementById('moveList');
  list.innerHTML = '<div class="move-list-empty">Henüz hamle yapılmadı</div>';
  document.getElementById('gameOverOverlay').classList.add('hidden');

  // Player bilgilerini ayarla
  setupPlayerCards();

  // Tahta yönü
  const orientation = (gameState.mode === 'human-black') ? 'black' : 'white';

  // chessboard.js config
  const config = {
    draggable: gameState.mode !== 'bot-vs-bot',
    position: 'start',
    orientation,
    pieceTheme: 'https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/img/chesspieces/wikipedia/{piece}.png',
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: onSnapEnd,
  };

  if (board) board.destroy();
  board = Chessboard('chessboard', config);

  // Hamle geçmişi sıfırla
  updateGameInfo('Oyun başladı', 1);
  updateEvalBar(0);
  highlightLastMove(null, null);

  // Bot vs bot kontrollerini göster
  if (gameState.mode === 'bot-vs-bot') {
    document.getElementById('botControls').style.display = 'flex';
    setupBotControls();
    setTimeout(() => doBotMove(), 800);
  } else {
    document.getElementById('botControls').style.display = 'none';
    // Siyah oynuyorsak AI ilk hamleyi yapsın
    if (gameState.mode === 'human-black') {
      setTimeout(() => doBotMove(), 800);
    }
  }
}

function setupPlayerCards() {
  const { mode, model1, model2 } = gameState;
  const m1 = MODEL_INFO[model1];
  const m2 = MODEL_INFO[model2];

  const topName    = document.getElementById('topPlayerName');
  const topModel   = document.getElementById('topPlayerModel');
  const topAvatar  = document.getElementById('topAvatar');
  const botName    = document.getElementById('bottomPlayerName');
  const botModel   = document.getElementById('bottomPlayerModel');
  const botAvatar  = document.getElementById('bottomAvatar');

  if (mode === 'human-white') {
    topAvatar.textContent  = m1.emoji;
    topName.textContent    = `${m1.label} (Siyah)`;
    topModel.textContent   = m1.tag;
    botAvatar.textContent  = '👤';
    botName.textContent    = 'Sen (Beyaz)';
    botModel.textContent   = 'İnsan';
  } else if (mode === 'human-black') {
    topAvatar.textContent  = '👤';
    topName.textContent    = 'Sen (Siyah)';
    topModel.textContent   = 'İnsan';
    botAvatar.textContent  = m1.emoji;
    botName.textContent    = `${m1.label} (Beyaz)`;
    botModel.textContent   = m1.tag;
  } else {
    topAvatar.textContent  = m2.emoji;
    topName.textContent    = `${m2.label} (Siyah)`;
    topModel.textContent   = m2.tag;
    botAvatar.textContent  = m1.emoji;
    botName.textContent    = `${m1.label} (Beyaz)`;
    botModel.textContent   = m1.tag;
  }
}

// ══════════════════════════════════════════════════════════════════════
// HUMAN MOVE HANDLING
// ══════════════════════════════════════════════════════════════════════

function onDragStart(source, piece) {
  if (game.game_over()) return false;
  if (!gameState.running) return false;
  // Human rengi kontrolü
  const humanColor = gameState.mode === 'human-black' ? 'b' : 'w';
  if (game.turn() !== humanColor) return false;
  if ((piece.search(/^b/) !== -1 && humanColor === 'w')) return false;
  if ((piece.search(/^w/) !== -1 && humanColor === 'b')) return false;
  return true;
}

function onDrop(source, target) {
  // Piyon terfisi kontrolü
  const promotion = isPromotion(source, target) ? 'q' : undefined;

  const move = game.move({
    from: source,
    to: target,
    promotion,
  });

  if (move === null) return 'snapback';

  processMove(move, false);

  // Bot sırası
  setTimeout(() => doBotMove(), 300);
}

function onSnapEnd() {
  board.position(game.fen());
}

function isPromotion(from, to) {
  const piece = game.get(from);
  if (!piece || piece.type !== 'p') return false;
  const toRank = parseInt(to[1]);
  return (piece.color === 'w' && toRank === 8) || (piece.color === 'b' && toRank === 1);
}

// ══════════════════════════════════════════════════════════════════════
// BOT MOVE
// ══════════════════════════════════════════════════════════════════════

async function doBotMove() {
  if (!gameState.running || gameState.paused) return;
  if (game.game_over()) { showGameOver(); return; }

  const isWhiteTurn = game.turn() === 'w';
  const { mode, model1, model2 } = gameState;

  // Sıranın bot'a ait olup olmadığını kontrol et
  const isBotTurn =
    mode === 'bot-vs-bot' ||
    (mode === 'human-white' && !isWhiteTurn) ||
    (mode === 'human-black' && isWhiteTurn);

  if (!isBotTurn) return;

  // Hangi model?
  let modelKey;
  if (mode === 'bot-vs-bot') {
    modelKey = isWhiteTurn ? model1 : model2;
  } else {
    modelKey = model1;
  }

  // Thinking göster
  const thinkSide = determineThinkingSide(isWhiteTurn);
  setThinking(thinkSide, true);
  updateGameInfo(`${MODEL_INFO[modelKey].emoji} ${MODEL_INFO[modelKey].label} düşünüyor...`, game.moveNumber());

  try {
    const result = await fetchMove(game.fen(), modelKey);

    if (result.game_over || result.error) {
      showGameOver();
      return;
    }

    // Hamleyi uygula
    const move = game.move({
      from: result.from,
      to: result.to,
      promotion: result.promotion || 'q',
    });

    if (!move) {
      showToast(`❌ Model geçersiz hamle döndürdü: ${result.uci}`, 'error');
      return;
    }

    processMove(move, true, result.value);

    // Bot vs Bot: devam et
    if (mode === 'bot-vs-bot' && !game.game_over()) {
      setTimeout(() => doBotMove(), gameState.moveDelay);
    }

  } catch (err) {
    showToast(`❌ Backend hatası: ${err.message}`, 'error');
    setStatus('Backend hatası', false);
  } finally {
    setThinking('top', false);
    setThinking('bottom', false);
  }
}

function determineThinkingSide(isWhiteTurn) {
  // board orientation: 'white' → beyaz altta
  const orient = gameState.mode === 'human-black' ? 'black' : 'white';
  if (orient === 'white') return isWhiteTurn ? 'bottom' : 'top';
  return isWhiteTurn ? 'top' : 'bottom';
}

// ══════════════════════════════════════════════════════════════════════
// PROCESS MOVE
// ══════════════════════════════════════════════════════════════════════

function processMove(move, isBot, evalValue = null) {
  board.position(game.fen());
  highlightLastMove(move.from, move.to);

  const moveNum = Math.ceil(game.history().length / 2);
  addMoveToHistory(game.history().length, move.san, move.color);

  if (evalValue !== null) {
    updateEvalBar(evalValue);
    document.getElementById('modelValueText').textContent =
      evalValue > 0 ? `+${evalValue.toFixed(3)}` : evalValue.toFixed(3);
  }

  const statusText = game.in_check()
    ? '⚠️ Şah!'
    : (isBot ? '👤 Senin sıran' : '🤖 AI düşünüyor...');

  updateGameInfo(statusText, moveNum);

  if (game.game_over()) {
    setTimeout(showGameOver, 600);
  }
}

// ══════════════════════════════════════════════════════════════════════
// GAME OVER
// ══════════════════════════════════════════════════════════════════════

function showGameOver() {
  gameState.running = false;

  const overlay = document.getElementById('gameOverOverlay');
  const icon    = document.getElementById('gameOverIcon');
  const title   = document.getElementById('gameOverTitle');
  const reason  = document.getElementById('gameOverReason');

  if (game.in_checkmate()) {
    const winner = game.turn() === 'w' ? 'Siyah' : 'Beyaz';
    icon.textContent  = '♟';
    title.textContent = `${winner} Kazandı!`;
    reason.textContent = 'Şah-Mat';
  } else if (game.in_stalemate()) {
    icon.textContent  = '🤝';
    title.textContent = 'Beraberlik';
    reason.textContent = 'Pat';
  } else if (game.in_draw()) {
    icon.textContent  = '🤝';
    title.textContent = 'Beraberlik';
    reason.textContent = '50 hamle / üç tekrar / yetersiz materyal';
  } else {
    icon.textContent  = '🏳';
    title.textContent = 'Oyun Bitti';
    reason.textContent = game.result?.() || '—';
  }

  overlay.classList.remove('hidden');
}

// ══════════════════════════════════════════════════════════════════════
// BOT VS BOT CONTROLS
// ══════════════════════════════════════════════════════════════════════

function setupBotControls() {
  const pauseBtn   = document.getElementById('pauseBtn');
  const speedRange = document.getElementById('speedRange');
  const speedVal   = document.getElementById('speedValue');

  pauseBtn.addEventListener('click', () => {
    gameState.paused = !gameState.paused;
    pauseBtn.textContent = gameState.paused ? '▶ Devam Et' : '⏸ Duraklat';
    pauseBtn.classList.toggle('paused', gameState.paused);
    if (!gameState.paused) doBotMove();
  });

  speedRange.addEventListener('input', () => {
    gameState.moveDelay = parseInt(speedRange.value);
    speedVal.textContent = (gameState.moveDelay / 1000).toFixed(1) + 's';
  });
}

// ══════════════════════════════════════════════════════════════════════
// NAVIGATION
// ══════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  // New game butonu
  document.getElementById('newGameBtn').addEventListener('click', () => {
    gameState.running = false;
    gameState.paused  = false;
    document.getElementById('gameScreen').classList.add('hidden');
    document.getElementById('setupScreen').classList.remove('hidden');
  });

  // Rematch
  document.getElementById('rematchBtn').addEventListener('click', () => {
    initGame();
  });

  initParticles();
  initSetupScreen();
  updateSetupUI();
  checkBackend();
});
