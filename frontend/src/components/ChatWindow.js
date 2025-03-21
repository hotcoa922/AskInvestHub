// src/components/ChatWindow.js
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './ChatWindow.css';

import aiAvatar from '../assets/ai.jpg';
import humanAvatar from '../assets/human.jpg';

const ChatWindow = () => {
  const [conversation, setConversation] = useState([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingTime, setLoadingTime] = useState(0);

  const chatWindowRef = useRef(null);
  const timerIdRef = useRef(null);
  const startTimeRef = useRef(null);

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [conversation]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setConversation((prev) => [
      ...prev,
      { sender: 'user', content: query }
    ]);

    setLoading(true);
    setLoadingTime(0);
    startTimeRef.current = Date.now();

    timerIdRef.current = setInterval(() => {
      const secondsElapsed = Math.floor((Date.now() - startTimeRef.current) / 1000);
      setLoadingTime(secondsElapsed);
    }, 1000);

    try {
      const response = await axios.post('http://localhost:8000/agent/ask', { query });
      const messages = response.data.messages;
      const aiRawContent = messages[messages.length - 1].content;
      const finalSeconds = Math.floor((Date.now() - startTimeRef.current) / 1000);

      let displayContent = aiRawContent;
      try {
        const parsed = JSON.parse(aiRawContent);
        if (parsed.output) {
          displayContent = parsed.output;
        }
      } catch (err) {}

      setConversation((prev) => [
        ...prev,
        { sender: 'ai', content: displayContent, finalTime: finalSeconds }
      ]);
    } catch (error) {
      console.error('API 호출 오류:', error);
      setConversation((prev) => [
        ...prev,
        { sender: 'ai', content: '서버와 통신 중 오류가 발생했습니다.' }
      ]);
    } finally {
      setLoading(false);
      clearInterval(timerIdRef.current);
      timerIdRef.current = null;
    }
    setQuery('');
  };

  return (
    <div className="chat-container-inner">
      <header className="chat-header">
        <h2>AIH</h2>
        <div className="header-question">?</div>
      </header>

      <div className="chat-window" ref={chatWindowRef}>
        {conversation.map((msg, index) => {
          const isUser = msg.sender === 'user';
          const avatar = isUser ? humanAvatar : aiAvatar;
          return (
            <div key={index} className={`chat-message ${isUser ? 'user' : 'ai'}`}>
              {!isUser && <img src={avatar} alt="avatar" className="avatar" />}
              <div className="message-bubble">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.content}
                </ReactMarkdown>
                {msg.sender === 'ai' && msg.finalTime != null && (
                  <button className="final-time-button">
                    최종 소요시간 - {msg.finalTime}초
                  </button>
                )}
              </div>
              {isUser && <img src={avatar} alt="avatar" className="avatar" />}
            </div>
          );
        })}

        {loading && (
          <div className="chat-message ai">
            <img src={aiAvatar} alt="avatar" className="avatar" />
            <div className="message-bubble loading-bubble">
              로딩 중... ({loadingTime}초)
              <div className="spinner" />
            </div>
          </div>
        )}
      </div>

      <form className="chat-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="질문을 입력하세요..."
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          보내기
        </button>
      </form>
    </div>
  );
};

export default ChatWindow;
