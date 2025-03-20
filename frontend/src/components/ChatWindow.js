import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
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

  // 새 메시지 추가 시 자동 스크롤
  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [conversation]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    // 사용자 메시지 추가
    setConversation((prev) => [
      ...prev,
      { sender: 'user', content: query }
    ]);

    // 로딩 상태 및 시간 초기화
    setLoading(true);
    setLoadingTime(0);
    startTimeRef.current = Date.now();

    // 1초마다 경과 시간 갱신
    timerIdRef.current = setInterval(() => {
      const secondsElapsed = Math.floor((Date.now() - startTimeRef.current) / 1000);
      setLoadingTime(secondsElapsed);
    }, 1000);

    try {
      // 백엔드 호출
      const response = await axios.post('http://localhost:8000/agent/ask', { query });
      const messages = response.data.messages;
      const aiRawContent = messages[messages.length - 1].content;

      // 응답 시점에서 최종 경과시간을 직접 계산 (정확도 향상)
      const finalSeconds = Math.floor((Date.now() - startTimeRef.current) / 1000);

      let displayContent = aiRawContent;
      try {
        // {"input":"...","output":"..."} 구조 가정
        const parsed = JSON.parse(aiRawContent);
        if (parsed.output) {
          displayContent = parsed.output;
        }
      } catch (err) {
        // JSON 파싱 실패 시 원본 그대로 사용
      }

      // AI 메시지에 finalTime을 저장
      setConversation((prev) => [
        ...prev,
        {
          sender: 'ai',
          content: displayContent,
          finalTime: finalSeconds
        }
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

    // 입력창 초기화
    setQuery('');
  };

  return (
    <div className="chat-container">
      {/* 헤더 영역 */}
      <header className="chat-header">
        <h2>AIH</h2>
        <div className="header-question">?</div>
      </header>

      {/* 채팅창 영역 */}
      <div className="chat-window" ref={chatWindowRef}>
        {conversation.map((msg, index) => {
          const isUser = msg.sender === 'user';
          const avatar = isUser ? humanAvatar : aiAvatar;
          return (
            <div key={index} className={`chat-message ${isUser ? 'user' : 'ai'}`}>
              {/* AI는 왼쪽, USER는 오른쪽 */}
              {!isUser && <img src={avatar} alt="avatar" className="avatar" />}
              <div className="message-bubble">
                {msg.content}
                {/* AI 메시지에만 최종 소요시간 버튼 표시 */}
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

        {/* 로딩 중 표시 */}
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

      {/* 입력 폼 */}
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
