// src/App.js
import React, { useState } from 'react';
import ChatWindow from './components/ChatWindow';
import Sidebar from './components/Sidebar';
import './App.css';

const App = () => {
  // welcomeState가 true이면 환영 화면을 보여주고, false가 되면 채팅 인터페이스와 사이드바를 함께 보여줌
  const [showWelcome, setShowWelcome] = useState(true);

  const handleStart = () => {
    setShowWelcome(false);
  };

  if (showWelcome) {
    return (
      <div className="welcome-container">
        <div className="welcome-card">
          <h1>환영합니다!</h1>
          <p>
            AIH 서비스에 오신 것을 환영합니다. 이 서비스는 투자 관련 질문에 대해
            법률 및 포트폴리오 분석 에이전트를 활용하여 최적의 답변을 제공합니다.
          </p>
          <button className="start-button" onClick={handleStart}>
            시작하기
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <Sidebar />
      <ChatWindow />
    </div>
  );
};

export default App;
