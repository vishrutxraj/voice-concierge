// Seeded demo callers (services/order-api/order_api/seed.py). Each one exists
// to exercise a different path through the agent, and the "try" line is what to
// say to see it -- every one of these was run against the real backend.
export interface DemoCaller {
  phone: string;
  label: string;
  note: string;
  /** What to say to see this caller's path through the agent. */
  try: string;
}

export const DEMO_CALLERS: DemoCaller[] = [
  {
    phone: "9990000001",
    label: "Three open orders",
    note: "Speaker, bedsheets + pillow covers, yoga mat",
    try: "“Tell me where my yoga mat order is”, or “where is my order” then “the third one”",
  },
  {
    phone: "9990000019",
    label: "Two open orders",
    note: "Laptop sleeve (in transit), wireless mouse (pending)",
    try: "“Where is my order”, then “the wireless mouse”",
  },
  {
    phone: "9990000020",
    label: "Two orders both pending",
    note: "Vase (out for delivery), bookshelf + cushion covers (pending)",
    try: "“Where is my order”, then “the pending one” — it has to narrow down",
  },
  {
    phone: "9990000002",
    label: "Failed delivery attempt",
    note: "Running shoes — the courier missed you",
    try: "“Where is my order”, “yes please”, “in 3 days evening”, then “yes”",
  },
  {
    phone: "9990000014",
    label: "Two failed attempts",
    note: "Monitor stand",
    try: "Same as above — reschedule after repeated failures",
  },
  {
    phone: "9990000004",
    label: "Rescheduled twice already",
    note: "Table lamp — at the reschedule limit",
    try: "“Reschedule to in 4 days evening”, then “yes” — refused, handed to a colleague",
  },
  {
    phone: "9990000009",
    label: "Fully booked slot",
    note: "Desk organiser",
    try: "“Reschedule to in 3 days morning”, then “yes” — offers other slots",
  },
  {
    phone: "9990000016",
    label: "Blackout date",
    note: "Rice cooker",
    try: "“Reschedule to in 5 days afternoon”, then “yes” — we don’t deliver that day",
  },
  {
    phone: "9990000007",
    label: "Out for delivery (Chennai)",
    note: "Filter coffee kit",
    try: "Change the address to another city — refused; the same city works",
  },
  {
    phone: "9990000013",
    label: "Cash on delivery, two items",
    note: "Backpack + water bottle, in transit",
    try: "“When will my order arrive”",
  },
  {
    phone: "9990000006",
    label: "Pending order",
    note: "Cricket bat — not shipped yet",
    try: "“Where is my order”",
  },
  {
    phone: "9990000003",
    label: "No open orders",
    note: "Their only order was already delivered",
    try: "“Where is my order” — goes to a colleague",
  },
];
