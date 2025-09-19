$(document).ready(function () {
    const $chatContainer = $('#chat-container');
    const $chatLauncher = $('#chat-launcher');
    const $closeChatButton = $('#close-chat-button');
    const $chatWindow = $('#chat-window');
    const $chatForm = $('#chat-form');
    const $messageInput = $('#message-input');
    const $sendButton = $('#send-button');

    // --- Toggle Chat ---
    $chatLauncher.on('click', function () {
        $chatContainer.toggleClass('hidden');
    });

    $closeChatButton.on('click', function () {
        $chatContainer.addClass('hidden');
    });

    $chatForm.on('submit', handleFormSubmit);

    // --- Form Submit Handler ---
    async function handleFormSubmit(e) {
        e.preventDefault();
        const userMessage = $.trim($messageInput.val());
        if (userMessage) {
            appendMessage(userMessage, 'user');
            $messageInput.val('');
            toggleFormState(true);
            appendTypingIndicator();
            try {
                const botResponse = await callGeminiAPI(userMessage);
                removeTypingIndicator();
                appendMessage(botResponse, 'bot');
            } catch (error) {
                console.error("Error calling API:", error);
                removeTypingIndicator();
                const friendlyErrorMessage = "Sorry, I'm having trouble connecting to the AI. Please check your internet connection or try again in a moment.";
                appendMessage(`${friendlyErrorMessage}<br><br><small class="text-white/70">Details: ${error.message}</small>`, 'bot', true);
            } finally {
                toggleFormState(false);
            }
        }
    }

    // --- Append Message ---
    function appendMessage(text, sender, isError = false) {
        const formattedHtml = marked.parse(text);
        const $messageWrapper = $('<div>').addClass('flex');
        const $messageBubble = $('<div>')
            .addClass('message-bubble p-3 rounded-lg max-w-xs md:max-w-md shadow')
            .html(formattedHtml);

        if (sender === 'user') {
            $messageWrapper.addClass('justify-end user-message');
        } else {
            $messageWrapper.addClass('justify-start bot-message');
            if (isError) $messageBubble.css('background-color', 'rgba(239, 68, 68, 0.7)');
        }

        $messageWrapper.append($messageBubble);
        $chatWindow.append($messageWrapper);
        $chatWindow.scrollTop($chatWindow.prop("scrollHeight"));
    }

    // --- Typing Indicator ---
    function appendTypingIndicator() {
        const $typingWrapper = $('<div>', { id: 'typing-indicator' })
            .addClass('flex justify-start');

        const $typingBubble = $('<div>')
            .addClass('p-3 rounded-lg shadow flex items-center space-x-1')
            .css('background-color', 'rgba(255, 255, 255, 0.2)')
            .html(`<span class="font-medium text-white/80">Typing</span>
                   <div class="dot-flashing"></div>
                   <div class="dot-flashing"></div>
                   <div class="dot-flashing"></div>`);

        if ($('#dot-flashing-style').length === 0) {
            $('<style>', { id: 'dot-flashing-style' }).html(`
                .dot-flashing { 
                    width: 5px; 
                    height: 5px; 
                    background-color: white; 
                    border-radius: 50%; 
                    animation: dotFlashing 1s infinite linear alternate; 
                } 
                .dot-flashing:nth-child(2) { animation-delay: 0.2s; } 
                .dot-flashing:nth-child(3) { animation-delay: 0.4s; } 
                @keyframes dotFlashing { 
                    0% { opacity: 0.5; } 
                    50%, 100% { opacity: 1; transform: scale(1.2); } 
                }
            `).appendTo('head');
        }

        $typingWrapper.append($typingBubble);
        $chatWindow.append($typingWrapper);
        $chatWindow.scrollTop($chatWindow.prop("scrollHeight"));
    }

    function removeTypingIndicator() {
        $('#typing-indicator').remove();
    }

    // --- Toggle Input State ---
    function toggleFormState(isDisabled) {
        $messageInput.prop('disabled', isDisabled);
        $sendButton.prop('disabled', isDisabled);
        $messageInput.attr('placeholder', isDisabled ? "Assistant is typing..." : "Ask a health question...");
        if (!isDisabled) $messageInput.focus();
    }

    // --- Gemini API Call ---
    async function callGeminiAPI(prompt) {
        const apiKey = "AIzaSyBUfST8o0mRQ_vqTd-7JxDjUfpgvcQrlzI";
        const systemInstruction = `You are "Healthify", an AI assistant for a disease prediction project. Your only purpose is to answer questions about the Healthify project and its capabilities.

            Healthify uses a trained model to predict four specific conditions from two types of data:

            1.  **Image-Based Predictions:**
                * **Pneumonia:** Predicted from chest X-ray images.
                * **Diabetic Retinopathy:** Predicted from retinal eye scan images.

            2.  **Tabular Data Predictions:**
                * **Heart Disease:** Predicted from tabular health data (e.g., cholesterol levels, blood pressure).
                * **Kidney Disease:** Predicted from tabular health data (e.g., blood test results).

            **Your Strict Rules:**
            - **Only discuss the four conditions listed above** (Pneumonia, Diabetic Retinopathy, Heart Disease, Kidney Disease) and how Healthify predicts them.
            - **Do not discuss any other diseases, treatments, or general health topics.** If asked about something else, state that it is "outside the scope of Healthify's current prediction models."
            - **You are a tool, not a doctor.** You must not provide medical advice or a diagnosis. Your predictions are for informational purposes only.
            - If a user asks for a diagnosis or seems to be in an emergency, you must firmly advise them to **consult a qualified healthcare professional immediately.**`;

        const chatHistory = [{ role: "user", parts: [{ text: systemInstruction + "\n\nUser Question: " + prompt }] }];
        const payload = { contents: chatHistory };
        const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;

        let retries = 3;
        let delay = 1000;

        for (let i = 0; i < retries; i++) {
            try {
                const response = await $.ajax({
                    url: apiUrl,
                    method: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify(payload)
                });

                if (response.candidates && response.candidates[0]?.content?.parts?.length > 0) {
                    return response.candidates[0].content.parts[0].text;
                } else {
                    console.error("Unexpected API response:", response);
                    throw new Error("Received an unexpected response from the AI service.");
                }
            } catch (error) {
                console.error(`Attempt ${i + 1} failed:`, error);
                if (i === retries - 1) throw error;
                await new Promise(res => setTimeout(res, delay));
                delay *= 2;
            }
        }
        throw new Error("API request failed after multiple retries.");
    }
});