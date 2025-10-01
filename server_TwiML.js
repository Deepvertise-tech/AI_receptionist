// ===== AI Receptionist — PSTN + Browser (TwiML App), OpenAI planner, SMTP message email =====
require('dotenv').config();
const express = require('express');
const twilio = require('twilio');
const OpenAI = require('openai');
const nodemailer = require('nodemailer');
const { BUSINESS, computeTotal } = require('./business');

const app = express();
app.use(express.urlencoded({ extended: true })); // Twilio posts form-encoded
app.use(express.json());

// ---- Serve Twilio Voice SDK (for browser client) ----
const twilioSdkPath = require.resolve('@twilio/voice-sdk/dist/twilio.min.js');
app.get('/sdk/twilio.min.js', (_req, res) => res.sendFile(twilioSdkPath));

// ---------- SMTP ----------
const mailer = nodemailer.createTransport({
  host: process.env.SMTP_HOST,
  port: Number(process.env.SMTP_PORT || 465),
  secure: String(process.env.SMTP_SECURE || 'true') === 'true',
  auth: { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS },
});
async function sendEmail(subject, text) {
  await mailer.sendMail({
    from: process.env.SMTP_USER,
    to: process.env.EMAIL_TO || process.env.SMTP_USER,
    subject,
    text,
  });
}

// ---------- OpenAI ----------
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ---------- Speech-only number normalization ----------
const WORD2DIGIT_EN = {
  zero:'0', oh:'0', o:'0', one:'1', two:'2', three:'3', four:'4', for:'4', five:'5',
  six:'6', seven:'7', eight:'8', nine:'9', ten:'10', plus:'+', dash:'-', minus:'-',
  space:'', dot:'.'
};
const WORD2DIGIT_DE = {
  null:'0', eins:'1', ein:'1', zwei:'2', drei:'3', vier:'4', fünf:'5', funf:'5',
  sechs:'6', sieben:'7', acht:'8', neun:'9', zehn:'10', plus:'+', minus:'-',
  bindestrich:'-', leerzeichen:'', punkt:'.'
};
function normalizePhoneFromSpeech(text, lang='en-US') {
  if (!text) return '';
  const t = text.toLowerCase().replace(/[,;:]/g,' ');
  const tokens = t.split(/\s+/);
  const map = lang.startsWith('de') ? WORD2DIGIT_DE : WORD2DIGIT_EN;
  let out = '';
  for (const tok of tokens) {
    if (/^\+?\d+([.\-\s]?\d+)*$/.test(tok)) { out += tok.replace(/\s/g,''); continue; }
    if (map[tok] != null) { out += map[tok]; continue; }
    const maybeDigits = tok.replace(
      /(one|two|three|four|for|five|six|seven|eight|nine|zero|oh|o)/g,
      m=>WORD2DIGIT_EN[m] ?? ''
    );
    if (/^\d+$/.test(maybeDigits)) { out += maybeDigits; continue; }
  }
  out = out.replace(/--+/g,'-').replace(/\.\.+/g,'.').replace(/(?!^)\+/g,'');
  return out;
}

// ---------- Detect German tokens to switch STT model only ----------
const GERMAN_TOKENS = [
  'straße','strasse','allee','platz','weg','ring','gasse','ufer','chaussee','hausnummer','postleitzahl','stadt',
  'eins','zwei','drei','vier','fünf','funf','sechs','sieben','acht','neun','zehn','uhr'
];
function looksGerman(text='') {
  const t = text.toLowerCase();
  return GERMAN_TOKENS.some(w => t.includes(w)) || /[äöüß]/.test(t);
}

// ---------- Dynamic ASR hints ----------
function buildHints(pipeline = null) {
  const addrDe = ['straße','strasse','allee','platz','weg','ring','gasse','ufer','chaussee','hausnummer','postleitzahl','stadt','uhr'];
  const numDe = ['null','eins','zwei','drei','vier','fünf','funf','sechs','sieben','acht','neun','zehn'];
  const common = ['yes','no','repeat','help','name','phone','number','address','cancel','nothing',"that's all",'no thanks','no thank you', ...addrDe, ...numDe];
  const menu = BUSINESS.menu.map(m => m.item);
  if (pipeline === 'order') return [...common, 'order','quantity','qty','delivery','pickup','takeaway','time','price', ...menu].join(',');
  if (pipeline === 'booking') return [...common, 'book','reservation','table','people','party','time','date','today','tomorrow'].join(',');
  if (pipeline === 'message') return [...common, 'leave a message','message','done','finish'].join(',');
  return ['book','reservation','table','order','leave a message','message','price','hours','menu', ...menu, ...common].join(',');
}

// ---------- Minimal per-call state ----------
const calls = new Map();
const getCall = (sid) => {
  if (!calls.has(sid)) {
    calls.set(sid, {
      context: {
        tts_lang: 'en-US',   // speak English by default
        stt_lang: 'en-US',   // recognize EN by default; dynamically flips to de-DE when German tokens detected
        active_pipeline: null,
        await_more: false,
        collect_message: false,
        silence_count: 0,
        turn_count: 0
      },
      history: [],
      lastSay: ''
    });
  }
  return calls.get(sid);
};

// ---------- TwiML helpers ----------
const say = (text, lang='en-US') => `<Say voice="Polly.Joanna" language="${lang}">${text}</Say>`;
const gather = (inner, opts = {}) => {
  const {
    action = '/voice/handle',
    language = 'en-US',           // STT language (dynamic)
    input = 'speech',
    speechTimeout = 'auto',
    actionOnEmptyResult = 'true',
    hints = buildHints(),
    speechModel = 'phone_call',
    profanityFilter = 'false',
    method = 'POST',
  } = opts;
  return `
    <Gather input="${input}" action="${action}" method="${method}"
            language="${language}" speechTimeout="${speechTimeout}"
            actionOnEmptyResult="${actionOnEmptyResult}"
            speechModel="${speechModel}" profanityFilter="${profanityFilter}"
            hints="${hints}">
      ${inner}
    </Gather>`;
};

// ---------- Stability ----------
const FAST_TIMEOUT_MS = 15000;
function withTimeout(promise, ms = FAST_TIMEOUT_MS) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error('LLM timeout')), ms))
  ]);
}
const MAX_TURNS = 30;

// ---------- Planner helpers ----------
function computeObjectiveAndMissing(ctx) {
  const inOrder = ctx.active_pipeline === 'order';
  const inBooking = ctx.active_pipeline === 'booking';
  const missing = [];
  if (inOrder) {
    if (!ctx.name) missing.push('name');
    if (!ctx.phone) missing.push('phone');
    if (!ctx.order_item) missing.push('order_item');
    if (!ctx.order_qty) missing.push('order_qty');
  }
  if (inBooking) {
    if (!ctx.name) missing.push('name');
    if (!ctx.phone) missing.push('phone');
    if (!ctx.service) missing.push('service');
    if (!ctx.when) missing.push('when');
  }
  const objective = inOrder ? 'order' : inBooking ? 'booking' : (ctx.active_pipeline === 'message' ? 'message' : 'idle');
  return { objective, missing };
}

function plannerSystemPrompt() {
  return `
You are the voice receptionist for "${BUSINESS.name}".
Speak English by default. Switch TTS to German only if asked. (Server may still use German STT for addresses.)
Reuse known caller info (name/phone/address) across pipelines; confirm briefly instead of re-asking.

Output STRICT JSON:
{
  "say": "string",
  "listen": true|false,
  "end_call": true|false,
  "hints": "comma,separated,stt,hints",
  "pipeline": "idle|order|booking|message",
  "pipeline_done": true|false,
  "switch_to": "order|booking|message|null",
  "offer_more": true|false,
  "action": "none|start_message|send_message|book_appointment|place_order|update_field",
  "fields": {
    "name":"...", "phone":"...", "address":"...",
    "service":"...", "when":"...",
    "order_item":"...", "order_qty":2,
    "message":"...", "computed_total": "number|null",
    "changes": { "field":"...", "value":"..." }
  }
}

Facts:
- Hours: ${BUSINESS.hours}
- Address: ${BUSINESS.address}
- Tables total: ${BUSINESS.tables_total}
- Policies: booking window ${BUSINESS.policies.booking_window_days} days; ${BUSINESS.policies.cancel_policy}
- Menu:
${BUSINESS.menu.map(m => `  - ${m.item}: $${m.price.toFixed(2)}`).join('\n')}

Flow:
- Pipelines: order / booking / message. If user asks to switch, set "switch_to".
- Collect only the missing fields. Keep replies short, natural.
- ORDER: name → phone → (address if delivery) → item → qty. Repeat phone/address to confirm; allow correction. Include/confirm total price.
- BOOKING: name → phone → party/service → when. Confirm phone; allow correction.
- MESSAGE: action="start_message", listen=true. Server captures next utterance, emails it, then we ask if anything else.
- After pipeline_done=true: offer more; if no, end_call=true with a warm thanks.
- If item not found: suggest 2–3 alternatives from menu.
- Low ASR confidence: ask a brief, targeted clarification.
- Output JSON only.`;
}

async function runPlanner({ call, userText, callerNumber }) {
  const sys = plannerSystemPrompt();
  const { objective, missing } = computeObjectiveAndMissing(call.context);
  const context = {
    caller: callerNumber || 'unknown',
    active_pipeline: call.context.active_pipeline || 'idle',
    objective,
    missing_fields: missing,
    await_more: !!call.context.await_more,
    asr_confidence: call.context.asr_confidence || 0,
    tts_lang: call.context.tts_lang || 'en-US',
    stt_lang: call.context.stt_lang || 'en-US',
    name: call.context.name || null,
    phone: call.context.phone || null,
    address: call.context.address || null,
    service: call.context.service || null,
    when: call.context.when || null,
    order_item: call.context.order_item || null,
    order_qty: call.context.order_qty || null,
    computed_total: call.context.computed_total ?? null,
  };

  const messages = [
    { role: 'system', content: sys },
    { role: 'user', content: `CONTEXT:\n${JSON.stringify(context, null, 2)}` },
    ...call.history.slice(-6),
    { role: 'user', content: userText?.trim() || '(no speech captured)' }
  ];

  const resp = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    temperature: 0.2,
    response_format: { type: 'json_object' },
    messages
  });

  let plan = {};
  try {
    plan = JSON.parse(resp.choices?.[0]?.message?.content || '{}');
  } catch {
    plan = { say: "Sorry, I didn’t catch that. Could you please repeat?", listen: true, pipeline: 'idle' };
  }
  return plan;
}

// ---------- Actions ----------
async function performAction(action, fields, call, req) {
  switch (action) {
    case 'start_message':
      call.context.collect_message = true;
      call.context.active_pipeline = 'message';
      call.context.await_more = false;
      break;
    case 'send_message': {
      const caller = req.body.From || fields?.phone || 'unknown';
      const subject = `New voice message from ${caller}`;
      const text = `From: ${caller}\n\nMessage:\n${(fields?.message || '').trim() || '(empty)'}\n\nContext:\n${JSON.stringify(call.context, null, 2)}`;
      sendEmail(subject, text).then(
        () => console.log('[action] send_message -> email sent'),
        (e) => console.error('Email send failed:', e.message)
      );
      break;
    }
    case 'book_appointment':
      console.log('[action] book_appointment ->', fields);
      break;
    case 'place_order':
      console.log('[action] place_order ->', fields);
      break;
    case 'update_field':
      console.log('[action] update_field ->', fields);
      break;
    default:
      break;
  }
}

// ---------- Health ----------
app.get('/', (_req, res) => res.send('AI Receptionist running (PSTN + Browser)'));

// ---------- Voice entry (used by both PSTN + Twilio Client) ----------
app.post('/voice/inbound', (req, res) => {
  const callSid = req.body.CallSid;
  const call = getCall(callSid);
  call.context.tts_lang = 'en-US';
  call.context.stt_lang = 'en-US';
  const welcome = `Hi! Welcome to ${BUSINESS.name}. How can I help you today?`;
  const twiml = `<Response>${
    gather(say(welcome, call.context.tts_lang), { hints: buildHints(null), language: call.context.stt_lang })
  }</Response>`;
  res.type('text/xml').send(twiml);
});

// ---------- Voice turn ----------
app.post('/voice/handle', async (req, res) => {
  const callSid = req.body.CallSid;
  const callerNumber = req.body.From || '';
  const speechRaw = (req.body.SpeechResult || '').trim();
  const confidence = parseFloat(req.body.Confidence || req.body.SpeechConfidence || '0') || 0;

  const call = getCall(callSid);
  const ctx = call.context;
  ctx.turn_count = (ctx.turn_count || 0) + 1;
  ctx.asr_confidence = confidence;

  // TTS language switch on user request
  if (speechRaw) {
    if (/\b(speak|switch|talk) (in )?german|auf deutsch\b/i.test(speechRaw)) ctx.tts_lang = 'de-DE';
    if (/\b(speak|switch|talk) (in )?english|auf englisch\b/i.test(speechRaw)) ctx.tts_lang = 'en-US';
  }

  // Next-turn STT language (dynamic)
  ctx.stt_lang = looksGerman(speechRaw) ? 'de-DE' : 'en-US';

  // Cap turns
  if (ctx.turn_count > MAX_TURNS) {
    const twiml = `<Response>${say('Thanks for calling. Goodbye!', ctx.tts_lang)}<Pause length="1"/><Hangup/></Response>`;
    calls.delete(callSid);
    return res.type('text/xml').send(twiml);
  }

  // Message capture mode (no LLM)
  if (ctx.collect_message) {
    const messageText = speechRaw || '(empty)';
    const subject = `New voice message from ${callerNumber || 'unknown'}`;
    const text = `From: ${callerNumber || 'unknown'}\n\nMessage:\n${messageText}\n\nContext:\n${JSON.stringify(ctx, null, 2)}`;
    sendEmail(subject, text).then(
      () => console.log('[message] emailed'),
      (e) => console.error('[message] email failed:', e.message)
    );
    ctx.collect_message = false;
    ctx.active_pipeline = null;
    ctx.await_more = true;
    const twiml = `<Response>${
      say(`Thanks. Your message has been sent. Is there anything else you’d like?`, ctx.tts_lang)
    }${gather(say('Go ahead.', ctx.tts_lang), { hints: buildHints(null), language: ctx.stt_lang })}</Response>`;
    return res.type('text/xml').send(twiml);
  }

  // Quick end while awaiting more
  const NEG_RE = /\b(no(thing)?( else)?|that'?s all|no thanks|no thank you|nah|nope)\b/i;
  if (ctx.await_more && speechRaw && NEG_RE.test(speechRaw)) {
    const twiml = `<Response>${say('Thanks for calling. Goodbye!', ctx.tts_lang)}<Hangup/></Response>`;
    calls.delete(callSid);
    return res.type('text/xml').send(twiml);
  }

  // Try to extract spoken phone digits
  try {
    const maybePhone = normalizePhoneFromSpeech(speechRaw, ctx.stt_lang);
    if (maybePhone && maybePhone.replace(/[^\d]/g,'').length >= 7) ctx.phone = maybePhone;
  } catch {}

  // LLM plan
  let plan;
  try {
    plan = await withTimeout(runPlanner({ call, userText: speechRaw, callerNumber }));
  } catch (e) {
    console.error('Planner timeout:', e.message);
    const twiml = `<Response>${gather(say(`Would you like to order food, book a table, or leave a message?`, ctx.tts_lang), { hints: buildHints(null), language: ctx.stt_lang })}</Response>`;
    return res.type('text/xml').send(twiml);
  }

  // Pipeline locking/switching
  if (plan.switch_to && ['order','booking','message'].includes(plan.switch_to)) {
    ctx.active_pipeline = plan.switch_to;
    ctx.await_more = false;
  }
  if (!ctx.active_pipeline && plan.pipeline && plan.pipeline !== 'idle') {
    ctx.active_pipeline = plan.pipeline;
  }
  const locked = ctx.active_pipeline || null;
  if (locked && plan.pipeline && plan.pipeline !== locked && !plan.switch_to) {
    plan.pipeline = locked;
    plan.listen = true;
    plan.end_call = false;
    plan.say = (plan.say ? `Let’s finish your ${locked} first. ` : `Let’s finish your ${locked} first. Please continue.`);
    if (plan.action && !['update_field','place_order','book_appointment','start_message','send_message','none'].includes(plan.action)) {
      plan.action = 'none';
    }
  }

  // Server-computed total for orders
  if ((plan.pipeline === 'order' || locked === 'order') && (ctx.order_item || plan.fields?.order_item)) {
    const item = plan.fields?.order_item || ctx.order_item;
    const qty = plan.fields?.order_qty ?? ctx.order_qty ?? 1;
    const total = computeTotal(item, qty);
    if (total !== null) {
      plan.fields = { ...(plan.fields || {}), computed_total: total };
      ctx.computed_total = total;
    }
  }

  // When pipeline done: unlock & offer more; announce total
  if (plan.pipeline_done === true) {
    if ((plan.pipeline === 'order' || locked === 'order') && (ctx.computed_total != null)) {
      const priceLine = `Total price: ${ctx.computed_total.toFixed(2)} euros. `;
      plan.say = plan.say ? (priceLine + plan.say) : priceLine;
    }
    ctx.active_pipeline = null;
    ctx.await_more = true;
    plan.end_call = false;
    if (!plan.say || !/anything else/i.test(plan.say)) {
      plan.say = (plan.say ? plan.say + ' ' : '') + 'Is there anything else you’d like?';
    }
    plan.listen = true;
  }

  // Persist fields
  if (plan?.fields && typeof plan.fields === 'object') {
    if (!plan.fields.phone && callerNumber) plan.fields.phone = callerNumber; // fallback
    Object.assign(ctx, plan.fields);
  }

  // History
  const assistantSayRaw = (plan?.say || "Okay.").trim();
  if (speechRaw) call.history.push({ role: 'user', content: speechRaw });
  call.history.push({ role: 'assistant', content: assistantSayRaw });
  if (call.history.length > 12) call.history.splice(0, call.history.length - 12);
  call.lastSay = assistantSayRaw;

  // Actions
  try {
    if (plan?.action && plan.action !== 'none') {
      await performAction(plan.action, plan.fields || {}, call, req);
    }
  } catch (e) {
    console.error('Action error:', e.message);
  }

  // Ending logic
  let mayEnd = false;
  if (ctx.await_more) {
    mayEnd = plan?.end_call === true;
    if (mayEnd) ctx.await_more = false;
  } else {
    mayEnd = plan?.end_call === true && !ctx.active_pipeline;
  }
  const BYE_RE = /\b(bye|goodbye|that'?s it|i'?m done|finished|thank you,? bye)\b/i;
  const userSpokeBye = speechRaw && BYE_RE.test(speechRaw);

  // Build response
  const nextSttLang = ctx.stt_lang;
  let twiml = '<Response>';
  if (mayEnd || userSpokeBye) {
    twiml += say('Thanks for calling. Goodbye!', ctx.tts_lang);
    twiml += '<Hangup/>';
    calls.delete(callSid);
  } else {
    const hints = buildHints(ctx.active_pipeline || plan.pipeline || (ctx.await_more ? null : null));
    twiml += say(assistantSayRaw, ctx.tts_lang);
    twiml += gather(say('Go ahead.', ctx.tts_lang), { hints, language: nextSttLang });
  }
  twiml += '</Response>';

  res.type('text/xml').send(twiml);
});

// ---------- Browser token (Twilio Client) ----------
app.get('/token', (req, res) => {
  const { AccessToken } = twilio.jwt;
  const { VoiceGrant } = AccessToken;

  const raw = (req.query.identity || '').trim();
  const identity = raw && /^[a-z0-9._-]{1,32}$/i.test(raw) ? raw : ('guest_' + Math.random().toString(36).slice(2,8));

  const token = new AccessToken(
    process.env.TWILIO_ACCOUNT_SID,
    process.env.TWILIO_API_KEY,
    process.env.TWILIO_API_SECRET,
    { identity }
  );

  token.addGrant(new VoiceGrant({
    outgoingApplicationSid: process.env.TWIML_APP_SID
  }));

  res.type('text/plain').send(token.toJwt());
});

// ---------- Shareable browser client ----------
app.get('/client', (_req, res) => {
  res.send(`<!doctype html>
<html>
<head><meta charset="utf-8"><title>AI Receptionist – Browser</title></head>
<body>
  <h3>AI Receptionist (Browser)</h3>
  <label>Display name: <input id="name" placeholder="e.g., bilal"/></label>
  <button id="connect">Connect</button>
  <button id="hangup" disabled>Hang up</button>
  <pre id="log" style="height:200px;overflow:auto;background:#f6f6f6;padding:8px"></pre>

  <script src="/sdk/twilio.min.js"></script>
  <script>
    const log = (...a)=>{ const el=document.getElementById('log'); el.textContent += a.join(' ')+"\\n"; el.scrollTop = el.scrollHeight; };
    let device, conn;

    async function getToken(identity){
      const url = identity ? '/token?identity=' + encodeURIComponent(identity) : '/token';
      const r = await fetch(url);
      if(!r.ok) throw new Error('Token HTTP '+r.status);
      return r.text();
    }

    document.getElementById('connect').onclick = async () => {
      try{
        const name = document.getElementById('name').value.trim();
        const jwt = await getToken(name);
        device = new Twilio.Device(jwt, { codecPreferences:['opus'], edge:'frankfurt' });
        device.on('registered', ()=>{ log('Device registered ✅'); });
        device.on('unregistered', ()=>log('Device unregistered'));
        device.on('error', e=> log('Device error ❌', e.message));
        device.on('connect', ()=>{ log('Connected 🔊'); document.getElementById('hangup').disabled=false; });
        device.on('disconnect', ()=>{ log('Disconnected 📴'); document.getElementById('hangup').disabled=true; });
        await device.register();
        conn = await device.connect(); // hits your TwiML App → /voice/inbound
      }catch(e){ log('Connect failed:', e.message); }
    };

    document.getElementById('hangup').onclick = () => { if(conn) conn.disconnect(); };
  </script>
</body>
</html>`);
});

// ---------- Start ----------
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log('Server on :' + PORT));
