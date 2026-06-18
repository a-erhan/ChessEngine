'use strict';

// ══════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ══════════════════════════════════════════════════════════════════════

// Backend URL: Sadece ayni sunucu uzerinden (Flask) relative path kullanilacak.
const API_BASE = '';

const MODEL_INFO = {
  agresif:  { label: 'Agresif',  tag: 'E10', class: 'dot-red' },
  tedbirli: { label: 'Tedbirli', tag: 'FT E5', class: 'dot-blue' },
  akil:     { label: 'Akil',     tag: 'Ultra E3', class: 'dot-gold' }
};

// ══════════════════════════════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════════════════════════════

let game = null;            // chess.js instance
let board = null;           // chessboard.js instance
let gameState = {
  mode: 'human-white',      // 'human-white' | 'human-black' | 'bot-vs-bot'
  model1: 'agresif',        // beyaz / tek model
  model2: 'tedbirli',       // siyah (bot vs bot'ta kullanilir)
  running: false,
  paused: false,
  moveDelay: 1200,
  moveHistory: [],
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
    toast.style.transform = 'translateX(16px)';
    toast.style.transition = '0.25s ease';
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
  const el = document.getElementById(`${side}Thinking`);
  const card = document.getElementById(`${side}PlayerCard`);
  if (!el || !card) return;
  if (thinking) {
    el.classList.remove('hidden');
    card.classList.add('active');
  } else {
    el.classList.add('hidden');
    card.classList.remove('active');
  }
}

function updateEvalBar(value) {
  // value: -1 to +1, +1 beyaz kazaniyor
  const pct = Math.round(((value + 1) / 2) * 100);
  const clamped = Math.max(5, Math.min(95, pct));
  document.getElementById('evalFill').style.height = clamped + '%';
  const whiteAdv = value > 0 ? `+${(value * 5).toFixed(1)}` : (value * 5).toFixed(1);
  document.getElementById('evalWhiteLabel').textContent = value > 0 ? whiteAdv : '0.0';
  document.getElementById('evalBlackLabel').textContent = value < 0 ? `+${(-value * 5).toFixed(1)}` : '0.0';
}

function addMoveToHistory(moveNum, san, color) {
  const list = document.getElementById('moveList');
  const empty = list.querySelector('.move-empty');
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
  document.querySelectorAll('.highlight-from, .highlight-to').forEach(el => {
    el.classList.remove('highlight-from', 'highlight-to');
  });

  if (!fromSq || !toSq) return;
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
// API
// ══════════════════════════════════════════════════════════════════════

async function checkBackend() {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      setStatus('Backend bagli', true);
      showToast('Chess Engine API baglandi!', 'success');
      return true;
    }
  } catch {
    setStatus('Backend baglanamadi', false);
    showToast('Backend cevrimdisi', 'error', 6000);
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
  document.querySelectorAll('.mode-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.mode-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      gameState.mode = card.dataset.mode;
      updateSetupUI();
    });
  });

  document.querySelectorAll('[data-slot="1"]').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('[data-slot="1"]').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      gameState.model1 = card.dataset.model;
    });
  });

  document.querySelectorAll('[data-slot="2"]').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('[data-slot="2"]').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      gameState.model2 = card.dataset.model;
    });
  });

  document.getElementById('startBtn').addEventListener('click', startGame);
}

function updateSetupUI() {
  const isBvB = gameState.mode === 'bot-vs-bot';
  document.getElementById('bot2Section').style.display = isBvB ? 'block' : 'none';

  const labels = {
    'human-white': 'AI Modeli Sec (Siyah Oynar)',
    'human-black': 'AI Modeli Sec (Beyaz Oynar)',
    'bot-vs-bot':  'Beyaz Taraf - 1. Model',
  };
  document.getElementById('modelSectionLabel').textContent = labels[gameState.mode];
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
  game = new Chess();
  gameState.running   = true;
  gameState.paused    = false;
  gameState.moveHistory = [];

  const list = document.getElementById('moveList');
  list.innerHTML = '<div class="move-empty">Henuz hamle yapilmadi</div>';
  document.getElementById('gameOverOverlay').classList.add('hidden');

  setupPlayerCards();

  const orientation = (gameState.mode === 'human-black') ? 'black' : 'white';

  const config = {
    draggable: gameState.mode !== 'bot-vs-bot',
    position: 'start',
    orientation,
    pieceTheme: 'img/chesspieces/wikipedia/{piece}.png',
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: onSnapEnd,
  };

  if (board) board.destroy();
  board = Chessboard('chessboard', config);

  updateGameInfo('Oyun basladi', 1);
  updateEvalBar(0);
  highlightLastMove(null, null);

  if (gameState.mode === 'bot-vs-bot') {
    document.getElementById('botControls').classList.remove('hidden');
    setupBotControls();
    setTimeout(() => doBotMove(), 800);
  } else {
    document.getElementById('botControls').classList.add('hidden');
    if (gameState.mode === 'human-black') {
      setTimeout(() => doBotMove(), 800);
    }
  }
}

function setupPlayerCards() {
  const { mode, model1, model2 } = gameState;
  const m1 = MODEL_INFO[model1];
  const m2 = MODEL_INFO[model2];

  const topColor    = document.getElementById('topColorBar');
  const topLabel    = document.getElementById('topPlayerLabel');
  const topName     = document.getElementById('topPlayerName');
  const topTag      = document.getElementById('topPlayerTag');

  const botColor    = document.getElementById('bottomColorBar');
  const botLabel    = document.getElementById('bottomPlayerLabel');
  const botName     = document.getElementById('bottomPlayerName');
  const botTag      = document.getElementById('bottomPlayerTag');

  if (mode === 'human-white') {
    topColor.className = 'player-color-bar black';
    topLabel.textContent = 'AI Siyah';
    topName.textContent = m1.label;
    topTag.textContent = m1.tag;

    botColor.className = 'player-color-bar white';
    botLabel.textContent = 'Sen Beyaz';
    botName.textContent = 'Insan';
    botTag.textContent = '';
  } else if (mode === 'human-black') {
    topColor.className = 'player-color-bar white';
    topLabel.textContent = 'Sen Siyah';
    topName.textContent = 'Insan';
    topTag.textContent = '';

    botColor.className = 'player-color-bar black';
    botLabel.textContent = 'AI Beyaz';
    botName.textContent = m1.label;
    botTag.textContent = m1.tag;
  } else {
    topColor.className = 'player-color-bar black';
    topLabel.textContent = 'AI Siyah';
    topName.textContent = m2.label;
    topTag.textContent = m2.tag;

    botColor.className = 'player-color-bar white';
    botLabel.textContent = 'AI Beyaz';
    botName.textContent = m1.label;
    botTag.textContent = m1.tag;
  }
}

// ══════════════════════════════════════════════════════════════════════
// HUMAN MOVE HANDLING
// ══════════════════════════════════════════════════════════════════════

function onDragStart(source, piece) {
  if (game.game_over()) return false;
  if (!gameState.running) return false;
  const humanColor = gameState.mode === 'human-black' ? 'b' : 'w';
  if (game.turn() !== humanColor) return false;
  if ((piece.search(/^b/) !== -1 && humanColor === 'w')) return false;
  if ((piece.search(/^w/) !== -1 && humanColor === 'b')) return false;
  return true;
}

function onDrop(source, target) {
  const promotion = isPromotion(source, target) ? 'q' : undefined;
  const move = game.move({
    from: source,
    to: target,
    promotion,
  });

  if (move === null) return 'snapback';

  processMove(move, false);
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

  const isBotTurn =
    mode === 'bot-vs-bot' ||
    (mode === 'human-white' && !isWhiteTurn) ||
    (mode === 'human-black' && isWhiteTurn);

  if (!isBotTurn) return;

  let modelKey;
  if (mode === 'bot-vs-bot') {
    modelKey = isWhiteTurn ? model1 : model2;
  } else {
    modelKey = model1;
  }

  const thinkSide = determineThinkingSide(isWhiteTurn);
  setThinking(thinkSide, true);
  const moveNum = Math.floor(game.history().length / 2) + 1;
  updateGameInfo(`${MODEL_INFO[modelKey].label} dusunuyor...`, moveNum);

  try {
    const result = await fetchMove(game.fen(), modelKey);

    if (result.game_over || result.error) {
      showGameOver();
      return;
    }

    const moveObj = { from: result.from, to: result.to };
    if (result.promotion) moveObj.promotion = result.promotion;
    // Satranc tahtasinda 1. veya 8. siraya gelen piyonlar icin otomatik q (vezir) terfisi ekle
    else if ((result.to.includes('8') || result.to.includes('1')) && game.get(result.from)?.type === 'p') {
      moveObj.promotion = 'q';
    }

    const move = game.move(moveObj);

    if (!move) {
      // Eger strict format calismazsa, uci string (orn 'e7e5') uzerinden deneme yapalim (chess.js 0.10.3)
      const fallbackMove = game.move(result.uci, { sloppy: true });
      if (!fallbackMove) {
        showToast(`Model gecersiz hamle dondurdu: ${result.uci}`, 'error');
        return;
      }
      processMove(fallbackMove, true, result.value);
    } else {
      processMove(move, true, result.value);
    }

    if (mode === 'bot-vs-bot' && !game.game_over()) {
      setTimeout(() => doBotMove(), gameState.moveDelay);
    }

  } catch (err) {
    showToast(`Backend hatasi: ${err.message}`, 'error');
    setStatus('Backend hatasi', false);
  } finally {
    setThinking('top', false);
    setThinking('bottom', false);
  }
}

function determineThinkingSide(isWhiteTurn) {
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
    ? 'Sah!'
    : (isBot ? 'Senin siran' : 'AI dusunuyor...');

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
  const resultDiv = document.getElementById('gameOverResult');
  const reasonDiv = document.getElementById('gameOverReason');

  if (game.in_checkmate()) {
    const winner = game.turn() === 'w' ? 'Siyah' : 'Beyaz';
    resultDiv.textContent = `${winner} Kazandi`;
    reasonDiv.textContent = 'Sah-Mat';
  } else if (game.in_stalemate()) {
    resultDiv.textContent = 'Beraberlik';
    reasonDiv.textContent = 'Pat';
  } else if (game.in_draw()) {
    resultDiv.textContent = 'Beraberlik';
    reasonDiv.textContent = '50 hamle / uc tekrar / yetersiz materyal';
  } else {
    resultDiv.textContent = 'Oyun Bitti';
    reasonDiv.textContent = game.result?.() || '—';
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

  // Event listener'ları birden fazla kez eklememek icin klonlayip degistiriyoruz
  const newPauseBtn = pauseBtn.cloneNode(true);
  pauseBtn.parentNode.replaceChild(newPauseBtn, pauseBtn);

  newPauseBtn.addEventListener('click', () => {
    gameState.paused = !gameState.paused;
    newPauseBtn.textContent = gameState.paused ? 'Devam Et' : 'Duraklat';
    newPauseBtn.classList.toggle('paused', gameState.paused);
    if (!gameState.paused) doBotMove();
  });

  const newSpeedRange = speedRange.cloneNode(true);
  speedRange.parentNode.replaceChild(newSpeedRange, speedRange);

  newSpeedRange.addEventListener('input', () => {
    gameState.moveDelay = parseInt(newSpeedRange.value);
    document.getElementById('speedValue').textContent = (gameState.moveDelay / 1000).toFixed(1) + 's';
  });
}

// ══════════════════════════════════════════════════════════════════════
// NAVIGATION
// ══════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('newGameBtn').addEventListener('click', () => {
    gameState.running = false;
    gameState.paused  = false;
    document.getElementById('gameScreen').classList.add('hidden');
    document.getElementById('setupScreen').classList.remove('hidden');
  });

  document.getElementById('rematchBtn').addEventListener('click', () => {
    initGame();
  });

  initSetupScreen();
  updateSetupUI();
  checkBackend();
});
