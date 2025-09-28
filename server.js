// ===== AI Receptionist — WEBHOOK-ONLY (no TwiML App, no /test) =====
require('dotenv').config();
const express = require('express');
const OpenAI = require('openai');
const nodemailer = require('nodemailer');
const { BUSINESS, computeTotal } = require('./business');

const app = express();
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

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

// ---------- Helpers (same logic as your current file) ----------
const WORD2DIGIT_EN = { zero:'0',oh:'0',o:'0',one:'1',two:'2',three:'3',four:'4',for:'4',five:'5',six:'6',seven:'7',eight:'8',nine:'9',ten:'10',plus:'+',dash:'-',minus:'-',space:'',dot:'.' };
const WORD2DIGIT_DE = { null:'0',eins:'1',ein:'1',zwei:'2',drei:'3',vier:'4',fünf:'5',funf:'5',sechs:'6',sieben:'7',acht:'8',neun:'9',zehn:'10',plus:'+',minus:'-',bindestrich:'-',leerzeichen:'',punkt:'.' };
function normalizePhoneFromSpeech(text, lang='en-US') {
  if (!text) return '';
  const t = text.toLowerCase().replace(/[,;:]/g,' ');
  const tokens = t.split(/\s+/);
  const map = lang.startsWith('de') ? WORD2DIGIT_DE : WORD2DIGIT_EN;
  let out = '';
  for (const tok of tokens) {
    if (/^\+?\d+([.\-\s]?\d+)*$/.test(tok)) { out += tok.replace(/\s/g,''); continue; }
    if (map[tok] != null) { out += map[tok]; continue; }
    const maybeDigits = tok.replace(/(one|two|three|four|for|five|six|seven|eight|nine|zero|oh|o)/g, m=>WORD2DIGIT_EN[m] ?? '');
    if (/^\d+$/.test(maybeDigits)) { out += maybeDigits; continue; }
  }
  return out.replace(/--+/g,'-').replace(/\.\.+/g,'.').replace(/(?!^)\+/g,'');
}
const GERMAN_TOKENS = ['straße','strasse','allee','platz','weg','ring','gasse','ufer','chaussee','hausnummer','postleitzahl','stadt','eins','zwei','drei','vier','fünf','funf','sechs','sieben','acht','neun','zehn','uhr'];
const looksGerman = (text='') => GERMAN_TOKENS.some(w => text.toLowerCase().includes(w)) || /[äöüß]/.test(text);
function buildHints(pipeline = null) {
  const addrDe = ['straße','strasse','allee','platz','weg','ring','gasse','ufer','chaussee','hausnummer','postleitzahl','stadt','uhr'];
  const numDe = ['null','eins','zwei','drei','vier','fünf','funf','sechs','sieben','acht','neun','zehn'];
  const common = ['yes','no','repeat','help','name','phone','number','address','cancel','nothing',"that's all",'no thanks','no thank you', ...addrDe, ...numDe];
  const menu = BUSINESS.menu.map(m => m.item);
  if (pipeline === 'order') return [...common, 'order','quantity','qty','delivery','pickup','time','price', ...menu].join(',');
  if (pipeline === 'booking') return [...common, 'book','reservation','table','people','party','time','date','today','tomorrow'].join(',');
  if (pipeline === 'message') return [...common, 'leave a message','message','done','finish'].join(',');
  return ['book','reservation','table','order','leave a message','message','price','hours','menu','pizza','pasta','salad', ...menu, ...common].join(',');
}

const calls = new Map();
const getCall = (sid) => {
  if (!calls.has(sid)) {
    calls.set(sid, {
      context: { tts_lang:'en-US', stt_lang:'en-US', active_pipeline:null, await_more:false, collect_message:false, asr_confidence:0, turn_count:0 },
      history: [], lastSay:''
    });
  }
  return calls.get(sid);
};

// TwiML helpers
const say = (text, lang='en-US') => `<Say voice="Polly.Joanna" language="${lang}">${text}</Say>`;
const gather = (inner, opts = {}) => {
  const { action='/voice/handle', language='en-US', input='speech', speechTimeout='auto', actionOnEmptyResult='true', hints=buildHints(), speechModel='phone_call', profanityFilter='false', method='POST' } = opts;
  return `<Gather input="${input}" action="${action}" method="${method}" language="${language}" speechTimeout="${speechTimeout}" actionOnEmptyResult="${actionOnEmptyResult}" speechModel="${speechModel}" profanityFilter="${profanityFilter}" hints="${hints}">${inner}</Gather>`;
};

// Timeouts
const FAST_TIMEOUT_MS = 15000;
const MAX_TURNS = 30;
function withTimeout(p, ms=FAST_TIMEOUT_MS){ return Promise.race([p, new Promise((_,rej)=>setTimeout(()=>rej(new Error('LLM timeout')),ms))]); }

// Planner (same as your version, trimmed)
function plannerSystemPrompt(){ return `You are the voice receptionist for "${BUSINESS.name}". Speak English by default... (prompt omitted here for brevity)`; }
async function runPlanner({ call, userText, callerNumber }) {
  const sys = plannerSystemPrompt();
  const messages = [
    { role: 'system', content: sys },
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
  try { plan = JSON.parse(resp.choices?.[0]?.message?.content || '{}'); }
  catch { plan = { say:"Sorry, I didn’t catch that.", listen:true, pipeline:'idle' }; }
  return plan;
}

// -------- Health
app.get('/', (_req, res) => res.send('Webhook mode running'));

// -------- INBOUND WEBHOOK (configure your Twilio number to POST here)
app.post('/voice/inbound', (req, res) => {
  const callSid = req.body.CallSid;
  const call = getCall(callSid);
  call.context.tts_lang = 'en-US';
  call.context.stt_lang = 'en-US';

  const welcome = `Hi! Welcome to ${BUSINESS.name}. How can I help you today?`;
  const twiml = `<Response>${
    gather(
      say(welcome, call.context.tts_lang),
      { hints: buildHints(null), language: call.context.stt_lang }
    )
  }</Response>`;

  res.type('text/xml').send(twiml);
});


app.post('/voice/handle', async (req, res) => {
  const callSid = req.body.CallSid;
  const callerNumber = req.body.From || '';
  const speechRaw = (req.body.SpeechResult || '').trim();
  const confidence = parseFloat(req.body.Confidence || '0') || 0;

  const call = getCall(callSid);
  const ctx = call.context;
  ctx.turn_count = (ctx.turn_count || 0) + 1;
  ctx.asr_confidence = confidence;

  // language handling
  ctx.stt_lang = looksGerman(speechRaw) ? 'de-DE' : 'en-US';

  // cap turns
  if (ctx.turn_count > MAX_TURNS) {
    const twiml = `<Response>${say('Thanks for calling. Goodbye!', ctx.tts_lang)}<Pause length="1"/><Hangup/></Response>`;
    calls.delete(callSid);
    return res.type('text/xml').send(twiml);
  }

  // phone normalization
  try {
    const maybe = normalizePhoneFromSpeech(speechRaw, ctx.stt_lang);
    if (maybe && maybe.replace(/[^\d]/g,'').length >= 7) ctx.phone = maybe;
  } catch {}

  // plan
  let plan;
  try { plan = await withTimeout(runPlanner({ call, userText: speechRaw, callerNumber })); }
  catch {
    const twiml = `<Response>${gather(say('Would you like to order food, book a table, or leave a message?', ctx.tts_lang), { hints: buildHints(null), language: ctx.stt_lang })}</Response>`;
    return res.type('text/xml').send(twiml);
  }

  // persist fields
  if (plan?.fields && typeof plan.fields === 'object') Object.assign(ctx, plan.fields);

  // compute totals if ordering
  if ((plan.pipeline === 'order') && (ctx.order_item || plan.fields?.order_item)) {
    const item = plan.fields?.order_item || ctx.order_item;
    const qty = plan.fields?.order_qty ?? ctx.order_qty ?? 1;
    const total = computeTotal(item, qty);
    if (total != null) { ctx.computed_total = total; plan.say = `Total price: ${total.toFixed(2)} euros. ` + (plan.say || ''); }
  }

  // send message action
  if (plan?.action === 'send_message') {
    const subject = `New voice message from ${callerNumber || ctx.phone || 'unknown'}`;
    const text = (plan?.fields?.message || speechRaw || '').trim();
    sendEmail(subject, text).catch(e=>console.error('Email fail:', e.message));
  }

  // build response
  const assistantSay = (plan?.say || 'Okay.').trim();
  if (speechRaw) call.history.push({ role:'user', content:speechRaw });
  call.history.push({ role:'assistant', content:assistantSay });
  if (call.history.length > 12) call.history.splice(0, call.history.length - 12);

  const hints = buildHints(plan.pipeline || null);
  const twiml = `<Response>${say(assistantSay, ctx.tts_lang)}${plan.end_call ? '<Hangup/>' : gather(say('Go ahead.', ctx.tts_lang), { hints, language: ctx.stt_lang })}</Response>`;
  if (plan.end_call) calls.delete(callSid);
  res.type('text/xml').send(twiml);
});

// -------- START
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log('Server on :' + PORT));
