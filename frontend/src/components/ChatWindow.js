import React, { useState } from 'react';
import axios from 'axios';
import './ChatWindow.css';

const ChatWindow = () => {
  // 대화 내역: { sender: 'user' | 'ai', content: string, finalTime?: number } 배열
  const [conversation, setConversation] = useState([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingTime, setLoadingTime] = useState(0);
  const [timerId, setTimerId] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    // 사용자 메시지 추가
    setConversation((prev) => [
      ...prev,
      { sender: 'user', content: query }
    ]);

    // 로딩 상태 초기화
    setLoading(true);
    setLoadingTime(0);

    // 1초마다 loadingTime을 1씩 증가
    const id = setInterval(() => {
      setLoadingTime((prev) => prev + 1);
    }, 1000);
    setTimerId(id);

    try {
      // API 호출
      const response = await axios.post('http://localhost:8000/agent/ask', { query });
      // 예) response.data.messages = [{ content: "{\"input\": \"...\", \"output\": \"...\"}" }, ...]
      const messages = response.data.messages;
      const aiRawContent = messages[messages.length - 1].content;

      // JSON 파싱 후 output 필드만 추출
      let displayContent = aiRawContent;
      try {
        const parsed = JSON.parse(aiRawContent);
        if (parsed.output) {
          displayContent = parsed.output;
        }
      } catch (err) {
        // JSON이 아니면 그대로 사용
      }

      // AI 메시지에 최종 소요 시간(finalTime)을 추가
      setConversation((prev) => [
        ...prev,
        { sender: 'ai', content: displayContent, finalTime: loadingTime }
      ]);
    } catch (error) {
      console.error('API 호출 오류:', error);
      setConversation((prev) => [
        ...prev,
        { sender: 'ai', content: '서버와 통신 중 오류가 발생했습니다.' }
      ]);
    } finally {
      setLoading(false);
      clearInterval(id);
      setTimerId(null);
    }

    setQuery('');
  };

  return (
    <div className="chat-container">
      <div className="chat-window">
        {conversation.map((msg, index) => (
          <div key={index} className={`chat-message ${msg.sender}`}>
            <div className="message-bubble">
              {msg.content}
            </div>
            {/* AI 메시지에 finalTime이 존재하면 3D 버튼 표시 */}
            {msg.sender === 'ai' && msg.finalTime != null && (
              <button className="final-time-button">
                최종 소요시간 - {msg.finalTime}초
              </button>
            )}
          </div>
        ))}

        {/* 로딩 중이면 스피너와 경과 시간 표시 */}
        {loading && (
          <div className="chat-message ai">
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
          전송
        </button>
      </form>
    </div>
  );
};

export default ChatWindow;
