// business.js — business profile & utilities

const BUSINESS = {
    name: "Bilal's Restaurant",
    phone: "+1 555 0100",
    hours: "Mon–Sat 10:00–22:00, Sun 12:00–20:00",
    address: "123 Sample Street, Berlin",
    tables_total: 20,
    menu: [
      { item: "margherita pizza", price: 9.5 },
      { item: "pepperoni pizza", price: 11.0 },
      { item: "pasta alfredo", price: 12.0 },
      { item: "caesar salad", price: 7.0 },
      { item: "tiramisu", price: 6.5 },
      { item: "Chicken Burger", price: 6.5 },
      { item: "Chicke Corn Soup", price: 4.5 },
      { item: "French Fries", price: 1.5 }
    ],
    policies: {
      booking_window_days: 14,
      cancel_policy: "Free cancellation up to 2 hours before time.",
    },
    faqs: [
      { q: "Do you have vegetarian options?", a: "Yes, several pizzas and salads are vegetarian." },
      { q: "Do you deliver?", a: "We do takeout; delivery via partner apps." },
    ],
  };
  
  // ---- Price helpers ----
  function findMenuItem(name = "") {
    const n = (name || "").toLowerCase().trim();
    return BUSINESS.menu.find(m => n.includes(m.item));
  }
  function computeTotal(itemName, qty) {
    const m = findMenuItem(itemName);
    const q = Number(qty || 1) || 1;
    return m ? +(m.price * q).toFixed(2) : null;
  }
  
  module.exports = { BUSINESS, findMenuItem, computeTotal };
  