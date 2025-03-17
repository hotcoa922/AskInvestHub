import React, { useState } from 'react';
import axios from 'axios';
import './ChatWindow.css';

const ChatWindow = () => {
  // 대화 내역은 { sender: 'user' | 'ai', content: string } 형식으로 저장합니다.
  const [conversation, setConversation] = useState([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    // 사용자의 질문을 대화 내역에 추가
    const userMessage = { sender: 'user', content: query };
    setConversation((prev) => [...prev, userMessage]);

    setLoading(true);
    try {
      // Docker에서 실행 중인 백엔드 주소에 맞게 URL을 설정 (예: localhost:8000)
      const response = await axios.post('http://localhost:8000/agent/ask', { query });
      // API 응답은 상태 객체로 반환되며, 그 안의 messages 배열에 AI 응답들이 담겨있습니다.
      // 이번 호출에 대한 새로운 AI 메시지는 messages 배열의 마지막 요소로 간주합니다.
      const messages = response.data.messages;
      const aiMessage = messages[messages.length - 1];
      // 대화 내역에 AI 메시지 추가 (aiMessage 객체의 content 속성을 사용)
      setConversation((prev) => [...prev, { sender: 'ai', content: aiMessage.content }]);
    } catch (error) {
      console.error('API 호출 오류:', error);
      setConversation((prev) => [...prev, { sender: 'ai', content: '서버와 통신 중 오류가 발생했습니다.' }]);
    }
    setQuery('');
    setLoading(false);
  };

  return (
    <div className="chat-container">
      <div className="chat-window">
        {conversation.map((msg, index) => (
          <div key={index} className={`chat-message ${msg.sender}`}>
            <div className="message-bubble">{msg.content}</div>
          </div>
        ))}
        {loading && (
          <div className="chat-message ai">
            <div className="message-bubble">로딩 중...</div>
          </div>
        )}
      </div>
      <form className="chat-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="질문을 입력하세요..."
        />
        <button type="submit" disabled={loading}>
          전송
        </button>
      </form>
    </div>
  );
};

export default ChatWindow;
