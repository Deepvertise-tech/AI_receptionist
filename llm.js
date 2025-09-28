// llm.js
const OpenAI = require("openai");
const business = require("./business");

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

function buildSystemPrompt() {
  const menuList = business.menu.map(m => `- ${m.item}: $${m.price.toFixed(2)}`).join("\n");
  const faqList = business.faqs.map(f => `Q: ${f.q}\nA: ${f.a}`).join("\n");

  return `
You are the AI receptionist for "${business.name}".
Speak concisely, friendly, and **voice-friendly** (short sentences).
When answering, rely ONLY on the facts below unless caller provides new info.

Business facts:
- Hours: ${business.hours}
- Total tables: ${business.tables_total}
- Policies: booking window ${business.policies.booking_window_days} days; ${business.policies.cancel_policy}

Menu (name and price):
${menuList}

FAQs:
${faqList}

Rules:
- If caller asks prices, hours, availability, or menu: answer directly from facts.
- If caller asks to book: collect name → service (or 'table for N') → desired date/time; then hand back to booking flow.
- Keep answers under 3 short sentences.
- If info is missing, say you’ll check and collect details (don’t invent).
`;
}

async function askLLM(userText) {
  const sys = buildSystemPrompt();

  const resp = await openai.chat.completions.create({
    model: "gpt-4o-mini",               // or another chat-capable model available to you
    temperature: 0.3,
    messages: [
      { role: "system", content: sys },
      { role: "user", content: userText }
    ]
  });

  return resp.choices?.[0]?.message?.content?.trim() || "Sorry, I didn’t catch that.";
}

module.exports = { askLLM };
