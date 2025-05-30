const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const chatBox = document.getElementById("chat-box");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  // Show user message
  const userMsg = document.createElement("div");
  userMsg.className = "user-message";
  userMsg.textContent = message;
  chatBox.appendChild(userMsg);
  input.value = "";

  // Show typing
  const botMsg = document.createElement("div");
  botMsg.className = "bot-message";
  botMsg.textContent = "🤔 Thinking...";
  chatBox.appendChild(botMsg);

  // Fetch backend response
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_query: message })
  });
  const data = await res.json();

  botMsg.textContent = data.response;
  chatBox.scrollTop = chatBox.scrollHeight;
});
