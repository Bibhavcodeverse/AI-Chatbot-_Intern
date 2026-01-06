let sessionId = null;
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const typingIndicator = document.getElementById('typing-indicator');
const suggestionsContainer = document.getElementById('suggestions-container');
const overlay = document.getElementById('overlay');

// Forms
const contactForm = document.getElementById('contact-form');
const feedbackForm = document.getElementById('feedback-form');
const goodbyeCard = document.getElementById('goodbye-card');

// Event Listeners
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

function appendMessage(text, isUser) {
    const div = document.createElement('div');
    div.classList.add('message');
    div.classList.add(isUser ? 'user-message' : 'bot-message');

    // Convert newlines to breaks for bot
    const formattedText = text.replace(/\n/g, '<br>');

    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    div.innerHTML = `
        <div class="message-content">${formattedText}</div>
        <div class="timestamp">${timestamp}</div>
    `;

    chatMessages.appendChild(div);
    scrollToBottom();
}

const scrollBtn = document.getElementById('scroll-bottom-btn');

scrollBtn.addEventListener('click', () => {
    scrollToBottom();
});

chatMessages.addEventListener('scroll', () => {
    const isNearBottom = chatMessages.scrollHeight - chatMessages.scrollTop - chatMessages.clientHeight < 100;
    if (isNearBottom) {
        scrollBtn.classList.remove('show');
    } else {
        scrollBtn.classList.add('show');
    }
});

function scrollToBottom() {
    // Use requestAnimationFrame for smoother timing with layout updates
    requestAnimationFrame(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

async function sendMessage() {
    // Check if running from file system
    if (window.location.protocol === 'file:') {
        appendMessage("⚠️ Error: You are opening the HTML file directly. Please access the chatbot via http://localhost:8000", false);
        return;
    }

    const text = userInput.value.trim();
    if (!text) return;

    // Clear input
    userInput.value = '';

    // Remove old suggestions
    suggestionsContainer.innerHTML = '';

    // Add User Message
    appendMessage(text, true);

    // Show typing
    typingIndicator.style.display = 'block';
    scrollToBottom();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text, session_id: sessionId })
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.status}`);
        }

        const data = await response.json();

        // Hide typing
        typingIndicator.style.display = 'none';

        // Update Session
        sessionId = data.session_id;

        // Add Bot Message
        appendMessage(data.response, false);

        // Handle Suggestions
        if (data.suggestions && data.suggestions.length > 0) {
            data.suggestions.forEach(suggestion => {
                const chip = document.createElement('div');
                chip.classList.add('suggestion-chip');
                chip.textContent = suggestion;
                chip.onclick = () => {
                    userInput.value = suggestion;
                    sendMessage();
                };
                suggestionsContainer.appendChild(chip);
            });
        }

        // Handle Session End
        if (data.session_ended) {
            userInput.disabled = true;
            sendBtn.disabled = true;
            setTimeout(() => {
                showContactForm();
            }, 1000);
        }

    } catch (error) {
        typingIndicator.style.display = 'none';
        appendMessage(`Sorry, something went wrong. (${error.message})`, false);
        console.error(error);
    }
}

// Form Handling
function showContactForm() {
    overlay.classList.add('active');
}

async function submitContact() {
    const name = document.getElementById('contact-name').value;
    const email = document.getElementById('contact-email').value;
    const phone = document.getElementById('contact-phone').value;

    if (!name || !email) {
        alert("Please fill in Name and Email.");
        return;
    }

    try {
        await fetch('/contact', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, email, phone })
        });

        // Switch to Feedback
        contactForm.classList.add('hidden');
        feedbackForm.classList.remove('hidden');
    } catch (e) {
        console.error(e);
        alert("Error saving details.");
    }
}

async function submitFeedback() {
    const feedback = document.getElementById('feedback-text').value;
    if (feedback) {
        await fetch('/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ feedback })
        });
    }
    finishSession();
}

function finishSession() {
    feedbackForm.classList.add('hidden');
    goodbyeCard.classList.remove('hidden');
}
