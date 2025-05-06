import { useEffect, useRef, useState } from 'react';
import { useImmer } from 'use-immer';
import './index.css';
import { useToggle } from "@uidotdev/usehooks";



function ToggleDemo({ on, toggle }) {
  return (
    <div>
      <label className="toggle">
        <input
          onChange={toggle}
          className="toggle-checkbox"
          type="checkbox"
          checked={on}
        />
        <div className="toggle-switch"></div>
        <span className="toggle-label">{on ? "On" : "Off"}</span>
      </label>
    </div>
  );
}


function ChatWindow({ messages }) {
    const bottomRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    return (
        <div className="chat-window">
            {messages.map((msg, index) => (
                <div
                    key={index} 
                    className={`message ${msg.role === 'user' ? 'user' : 'assistant'}`}
                >
                    {msg.content}
                </div>
            ))}
            <div ref={bottomRef} />
        </div>
    );
}

function ChatInput({ newMessage, setNewMessage, onSend }) {
    return (
        <div className="chat-input-container">
            <input
                type="text"
                value={newMessage}
                placeholder="Type your message..."
                onChange={(e) => setNewMessage(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && onSend()}
            />
            <button onClick={onSend}>Send</button>
        </div>
    );
}

function Chatbot() {
    const [messages, setMessages] = useImmer([]);
    const [newMessage, setNewMessage] = useState('');
    const [newdeepresearch, setdeepresearch] = useState(false);
    const [on, toggle] = useToggle(true);

    const sendMessage = async () => {
        if (!newMessage.trim()) return;

        // Add user message
        setMessages(draft => {
            draft.push({ role: 'user', content: newMessage });
        });
        setNewMessage('');
        try {
          const res = await fetch("http://localhost:5000/api/chat", {
              method:'POST',
              headers: {'Content-Type':'application/json'},
              body : JSON.stringify({message : newMessage})
          }); 
          const data = await res.json()
          setMessages(draft => {
            draft.push({'role': 'assistant', content: data.response});
          });
      } catch(err) {
        setMessages(draft => {
          draft.push({role :'assistant', content : 'Error reaching backend'})
        });
        console.error(err);
      }

    };

    return (
        <div className="chatbot-container">
            <div className="chat-header">MedHive AI Chatbot</div>
            <ChatWindow messages={messages} />
            <ChatInput
                newMessage={newMessage}
                setNewMessage={setNewMessage}
                onSend={sendMessage}
                newdeepresearch = {newdeepresearch}
            />
            <ToggleDemo toggle={toggle} on ={on} />
        </div>
    );
}

export default Chatbot;
